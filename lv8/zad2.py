import keras.datasets
import keras.datasets.mnist
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import models

model = models.load_model('LV8_model.keras')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

num_classes = 10
y_test_s = keras.utils.to_categorical(y_test, num_classes)
x_test_vector = x_test_s.reshape(-1, 784)

predictions = model.predict(x_test_vector)

for i in range(300):
    if y_test[i] != predictions[i].argmax():
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"Stvarna oznaka: {y_test[i]}, Predvidjena oznaka: {predictions[i].argmax()}")
        plt.show()
