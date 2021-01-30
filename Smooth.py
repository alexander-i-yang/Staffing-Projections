import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import numpy as np

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()


def tf_smooth(
        observation_index_points_,
        observations_,
        print_progress=True,
        num_iters=350,
        learning_rate=0.01
):
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
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Store the likelihood values during training, so we can plot the progress
    lls_ = np.zeros(num_iters, np.float64)
    for i in range(num_iters):
        if print_progress:
            percent = i * 100 / num_iters
            if percent % 10 == 0: print("%i%% done." % percent)

        with tf.GradientTape() as tape:
            loss = -target_log_prob(amplitude_var, length_scale_var,
                                    observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        lls_[i] = loss

    num_training_points = len(observation_index_points_)
    predictive_index_points_ = np.linspace(0, num_training_points, num_training_points, dtype=np.float64)
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=optimized_kernel,
        index_points=predictive_index_points_,
        observation_index_points=observation_index_points_,
        observations=observations_,
        observation_noise_variance=observation_noise_variance_var,
        predictive_noise_variance=0.)

    samples = gprm.sample(num_training_points)
    # Return actual values and log loss
    return samples[0, :], lls_
