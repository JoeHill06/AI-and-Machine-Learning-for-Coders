import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print(f"Here is what I learned: {l0.get_weights()}")