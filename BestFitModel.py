import math
import time

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

def raw_to_agents(talk_time_, occupancy_):
    def convert(talk_time_per_half_hour, occupancy):
        return talk_time_per_half_hour/occupancy/60
    ret = []
    for x, y in zip(talk_time_, occupancy_):
        ret.append(convert(x, y))
    return ret


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def round_threshold(num, threshold):
    return round_half_up(num+(0.5-threshold))

def round_threshold_list(nums, threshold=0.3):
    return [round_threshold(x, threshold) for x in nums]

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).


rawdf = pd.read_csv("inbound_monthly.csv")

observation_index_points_ = rawdf['interval'].astype(float)
observation_index_points_ = [[x] for x in observation_index_points_]
NUM_TRAINING_POINTS = len(observation_index_points_)
observations_ = np.asarray(rawdf['call_time'].astype(float))

print("obs:")
print(observations_)

# TODO: replace fake_occupancy_ with real occupancy data
fake_occupancy_ = np.linspace(0.75, 0.75, NUM_TRAINING_POINTS)

observations_ = raw_to_agents(observations_, fake_occupancy_)
print(observations_)

def build_gp(amplitude, length_scale, observation_noise_variance):
    """Defines the conditional dist. of GP outputs, given kernel parameters."""

    # Create the covariance kernel, which will be shared between the prior (which we
    # use for maximum likelihood training) and the posterior (which we use for
    # posterior predictive sampling)
    kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

    # Create the GP prior distribution, which we will use to train the model
    # parameters.
    return tfd.GaussianProcess(
        kernel=kernel,
        index_points=observation_index_points_,
        observation_noise_variance=observation_noise_variance)


gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})

x = gp_joint_model.sample()
lp = gp_joint_model.log_prob(x)

print("sampled {}".format(x))
print("log_prob of sample: {}".format(lp))

# Create the trainable model parameters, which we'll subsequently optimize.
# Note that we constrain them to be strictly positive.

# noinspection DuplicatedCode
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in
                       [amplitude_var,
                        length_scale_var,
                        observation_noise_variance_var]]


# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude, length_scale, observation_noise_variance):
    return gp_joint_model.log_prob({
        'amplitude': amplitude,
        'length_scale': length_scale,
        'observation_noise_variance': observation_noise_variance,
        'observations': observations_
    })


# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
print("Starting traning.")
for i in range(num_iters):
    if (i%100 == 0): print("We're %i%% of the way there!" % (i*100/num_iters))
    with tf.GradientTape() as tape:
        loss = -target_log_prob(amplitude_var, length_scale_var,
                                observation_noise_variance_var)
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

# Plot the loss evolution
plt.plot(lls_)
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.show()

# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.
predictive_index_points_ = np.linspace(0, NUM_TRAINING_POINTS, NUM_TRAINING_POINTS, dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# Create op to draw  50 independent samples, each of which is a *joint* draw
# from the posterior at the predictive_index_points_. Since we have 200 input
# locations as defined above, this posterior distribution over corresponding
# function values is a 200-dimensional multivariate Gaussian distribution!
num_samples = NUM_TRAINING_POINTS
samples = gprm.sample(num_samples)

# Plot the true function, observations, and posterior samples.
plt.figure(figsize=(12, 4))
plt.plot([x[0] for x in observation_index_points_], observations_,
            label='Raw Agent Predictions')
first_sample = samples[0, :]
# first_sample = round_threshold_list(first_sample)
# plt.plot(predictive_index_points_, samples[0, :], c='r', alpha=.1,
#              label='Smoothed Agent Predictions')
for i in range(num_samples):
    plt.plot(predictive_index_points_, samples[i, :], c='r', alpha=.1,
             label='Smoothed Agent Predictions' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.show()

print("done!")

print(first_sample)