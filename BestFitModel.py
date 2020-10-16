import math
import time
import datetime

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import os

import pymc3 as pm
from theano import shared
from pymc3.distributions.timeseries import GaussianRandomWalk

TRAIN_TF = False
SHOW_GRAPHS = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def raw_to_agents(talk_time_, occupancy_):
    def convert(talk_time_per_half_hour, occupancy):
        return (talk_time_per_half_hour / 3600) / occupancy

    ret = []
    for x, y in zip(talk_time_, occupancy_):
        ret.append(convert(x, y))
    return ret


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def round_threshold(num, threshold):
    return round_half_up(num + (0.5 - threshold))


def round_threshold_list(nums, threshold=0.3):
    return [int(max(round_threshold(x, threshold), 0)) for x in nums]


def get_data_by_program(rawdf, program_id):
    return (rawdf.loc[rawdf.program_id == program_id])


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


def smooth_model(x, y):
    LARGE_NUMBER = 1e5

    model = pm.Model()
    with model:
        smoothing_param = shared(0.9)
        mu = pm.Normal("mu", sigma=LARGE_NUMBER)
        tau = pm.Exponential("tau", 1.0 / LARGE_NUMBER)
        z = GaussianRandomWalk("z",
                               mu=mu,
                               tau=tau / (1.0 - smoothing_param),
                               shape=y.shape)
        obs = pm.Normal("obs",
                        mu=z,
                        tau=tau / smoothing_param,
                        observed=y)

    def infer_z(smoothing):
        with model:
            smoothing_param.set_value(smoothing)
            res = pm.find_MAP(vars=[z], method="L-BFGS-B")
            return res['z']

    smoothing = 0.95
    z_val = infer_z(smoothing)

    return z_val


def tf_train(observation_index_points, observations_, program_id):
    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp,
    })

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
    num_iters = 350
    optimizer = tf.optimizers.Adam(learning_rate=.01)

    # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)
    for i in range(num_iters):
        # if (i % 100 == 0): print("We're %i%% of the way there!" % (i * 100 / num_iters))
        with tf.GradientTape() as tape:
            loss = -target_log_prob(amplitude_var, length_scale_var,
                                    observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        lls_[i] = loss

    # print('Trained parameters:')
    # print('amplitude: {}'.format(amplitude_var._value().numpy()))
    # print('length_scale: {}'.format(length_scale_var._value().numpy()))
    # print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

    # Plot the loss evolution
    plt.plot(lls_)
    plt.title("Log loss for %i" % program_id)
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

    num_samples = NUM_TRAINING_POINTS
    samples = gprm.sample(num_samples)

    return samples[0, :]


tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

# Generate training data with a known noise level (we'll later try to recover
# this value from the data).

# I'm reading from csv because it's faster than reading from bigquery
# I don't want to read from bigquery every time I want to run this program
# See Bigquery.py for the method to import bigquery data into a csv
rawdf = pd.read_csv("inbound_monthly.csv", sep=r'\s*,\s*').drop("Unnamed: 0", axis=1)

rawdf['program_id'] = rawdf['program_id'].fillna(0)
rawdf = rawdf.astype({'program_id': 'int32'})
rawdf['time_interval'] = pd.to_datetime(rawdf['time_interval'])
program_ids = rawdf['program_id'].unique().tolist()
rawdf.sort_values(by=["program_id", "time_interval"], inplace=True)
rawdf.set_index(keys=["program_id"], drop=False, inplace=True)

timestamps = rawdf['time_interval'].unique().to_numpy()
timestamps.sort()
start_timestamp = timestamps[0]
end_timestamp = timestamps[-1]
points_per_id = int((end_timestamp - start_timestamp).total_seconds() / (30 * 60))

START_TIME = time.time()

final_df = pd.DataFrame()

for counter in range(len(program_ids)):
    # if counter > 3: break
    cur_id = program_ids[counter]
    if counter % 10 == 1:
        print("%i: %i/%i" % (cur_id, counter, len(program_ids)))
    counter += 1
    program_df = get_data_by_program(rawdf, cur_id)

    rows_to_add = []
    for i in range(points_per_id + 1):
        cur_timestamp = start_timestamp + datetime.timedelta(0, i * 30 * 60)
        if cur_timestamp not in program_df.values:
            rows_to_add.append([cur_timestamp, cur_id, 0, 0])
    df_to_add = pd.DataFrame(rows_to_add, columns=program_df.columns)
    program_df = program_df.append(df_to_add, ignore_index=True)
    program_df.sort_values(by="time_interval", inplace=True, ignore_index=True)

    NUM_TRAINING_POINTS = len(program_df)
    observation_index_points_ = range(0, NUM_TRAINING_POINTS)
    observation_index_points_ = [[float(x)] for x in observation_index_points_]
    observations_ = np.asarray(program_df['call_time'].astype(float))

    # TODO: replace fake_occupancy_ with real occupancy data
    fake_occupancy_ = np.linspace(0.75, 0.75, NUM_TRAINING_POINTS)
    observations_ = raw_to_agents(observations_, fake_occupancy_)

    processed_df = pd.DataFrame()
    time_index_ = program_df['time_interval']
    processed_df['Time'] = time_index_
    processed_df['Raw Agents'] = observations_
    processed_df['Program Id'] = [cur_id for x in range(NUM_TRAINING_POINTS)]

    smooth_data = smooth_model(np.asarray(observation_index_points_), np.asarray(observations_))
    processed_df['Smooth Rounded Data'] = round_threshold_list(smooth_data)

    if TRAIN_TF:
        tf_data = tf_train(observation_index_points_, observations_, cur_id)
        processed_df['TF Rounded Data'] = round_threshold_list(tf_data)

    final_df = final_df.append(processed_df, ignore_index=True)

    if SHOW_GRAPHS:
        plt.figure(figsize=(12, 4))
        plt.title("Data for %i" % cur_id)
        plt.plot(time_index_, processed_df['Raw Agents'], label="Raw Agents")
        plt.plot(time_index_, processed_df['Smooth Rounded Data'], label="Smoothed Rounded Agents")
        if TRAIN_TF: plt.plot(time_index_, processed_df['TF Rounded Data'], label="TF Rounded Agents")
        leg = plt.legend(loc='upper right')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.show()

print("done in %i seconds!" % (time.time() - START_TIME))
f = open("demofile2.txt", "a")
f.write(final_df.to_csv(index=False))
f.close()
