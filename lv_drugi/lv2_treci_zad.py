import numpy as np
import matplotlib . pyplot as plt
import matplotlib . image as img


img = plt.imread('road.jpg') #plt.imread('road.jpg')

img = img [ :,:,0]. copy ()


brightened_image = np.clip(img.astype(int) + 50).astype(np.uint8)


height, width, _ = image.shape
quarter_width = width // 4
cropped_image = image[:, quarter_width:2*quarter_width]


rotated_image = np.rot90(image)

flipped_image = np.flip(image, axis=1)

cv2.imshow('Originalna slika', image)
cv2.imshow('Posvijetljena slika', brightened_image)
cv2.imshow('Druga četvrtina slike', cropped_image)
cv2.imshow('Rotirana slika', rotated_image)
cv2.imshow('Zrcaljena slika', flipped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()