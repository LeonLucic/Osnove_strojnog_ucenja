from PIL import Image
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import models
from keras import utils as image
model = models.load_model('LV8_model.keras')





image = Image.open("example1.png")
image = image.convert("L")
image = image.resize((28, 28))

image_array = np.array(image)
image_array = image_array.reshape(1, 784)
image_array = image_array.astype("float32") / 255

predictions = model.predict(image_array)
print(predictions)
predicted_class = np.argmax(predictions, axis=1)


plt.imshow(image)
plt.title(f'Stvarni broj: 9, predikcija: {predicted_class[0]}')
plt.show()
