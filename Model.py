import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import CreateData as cdata
import math
import tensorflow.keras as keras
import matplotlib.pyplot as plt

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')

def bell_curve(layer):
    def ret(y_actual, y_pred):
        print(y_actual)
        print(y_pred)
        return 0
        # return kb.square(y_actual - y_pred)

    return ret


df = cdata.generate_bimodal_dataframe(24)
x = [df.hour]
y = df.call_volume

model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

history = model.fit(x, y, epochs=500, verbose=0)
rmse = list(map(math.sqrt, history.history['loss']))
plt.plot(rmse)
plt.show()
print("Final RMSE: %f" % (rmse[-1]))
# Make predictions.
x_tst = np.arange(0, 24, 0.1)
yhat = model(x_tst)
plt.plot(x_tst, yhat.numpy(), label="ree")
plt.plot(x_tst, cdata.bimodal_np(x_tst), label="hello")
plt.scatter(x, y)
plt.legend(loc="upper left")
plt.show()
