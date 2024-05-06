import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img[:,:,0].copy()

plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Originalna slika")
plt.show()

plt.figure()
plt.imshow(img, alpha=0.7, cmap="gray")
plt.title("Osvijetljena slika")
plt.show()

width, height = img.shape
first_quarter_width = width // 4
second_quarter_width = width // 2
second_quater_width_img = img[:,first_quarter_width:second_quarter_width]
plt.imshow(second_quater_width_img, cmap="gray")
plt.title("Druga četvrtina slike po širini")
plt.show()

rotated_img = np.rot90(img, k=3)
plt.imshow(rotated_img, cmap="gray")
plt.title("Rotirana slika za 90 clockwise")
plt.show()

mirrored_img = np.flip(img, axis=1)
plt.imshow(mirrored_img, cmap="gray")
plt.title("Zrcaljena slika")
plt.show()