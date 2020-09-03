import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import keras.backend as kb
import math
import CreateData as cdata
import tensorflow.keras as keras
import matplotlib.pyplot as plt

tfd = tfp.distributions
keras.backend.set_floatx('float64')

NUM_DAILY_DATA_POINTS = 24

x, y = cdata.generate_bimodal_sets(30, NUM_DAILY_DATA_POINTS)

@tf.function
def custom_loss(y_true, coeffs):
    a = tf.slice(coeffs, [0, 0], [1, 1])
    b = tf.slice(coeffs, [0, 1], [1, 1])
    c = tf.slice(coeffs, [0, 2], [1, 1])
    x_vals = tf.constant(np.array(range(NUM_DAILY_DATA_POINTS)).astype(np.float64))
    y_pred = a * kb.exp(b * kb.square(x_vals - c))
    return kb.mean(kb.square(y_pred - y_true), axis=-1)


def baseline_model():
    # create model
    i = keras.Input(shape=(NUM_DAILY_DATA_POINTS))
    x = keras.layers.Dense(64, kernel_initializer='glorot_uniform', activation='relu')(i)
    y = keras.layers.Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    o = keras.layers.Dense(NUM_DAILY_DATA_POINTS, kernel_initializer='normal', activation='linear')(y)
    model = keras.Model(i, o)
    model.compile(loss=custom_loss, optimizer="adam")
    return model


model = baseline_model()

history = model.fit(
    tf.convert_to_tensor(y),
    tf.convert_to_tensor(y),
    batch_size=32,
    epochs=500,
    verbose=0
)
rmse = list(map(math.sqrt, history.history['loss']))
print("Final RMSE: %f" % rmse[-1])
plt.plot(rmse)
plt.show()


def bell_curve(x, a, b, c):
    # print(x)
    return a * math.exp(b * (x - c) ** 2)


def bell_curve_np(x, a, b, c):
    return [bell_curve(i, a, b, c) for i in list(x)]


# Make predictions.
x_tst = np.arange(0, 24, 1)
yhat = model(tf.convert_to_tensor(y))
arr = yhat.numpy()[0]
a = arr[0]
b = arr[1]
c = arr[2]

x_eval = np.arange(0, 24, 0.1)
y_eval = bell_curve_np(x_eval, a, b, c)
print(a, b, c)

plt.scatter(x_tst, y[0], label="ree")
plt.plot(x_eval, bell_curve_np(x_eval, a, b, c), label="hello")
plt.legend(loc="upper left")
plt.show()
