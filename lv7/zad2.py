import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_5.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

print(f"Broj boja na originalnoj slici: {len(np.unique(img_array_aprox, axis = 0))}")
km = KMeans (n_clusters =5, init ='k-means++', n_init =5 , random_state =0 )
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

for i in range (len(km.cluster_centers_)):
    img_array_aprox[labels == i] = km.cluster_centers_[i]

img_new = np.reshape(img_array_aprox, (w,h,d))
img_new = (img_new*255).astype(np.uint8)

plt.figure()
plt.title("Kvantizirana slika")
plt.imshow(img_new)
plt.tight_layout()
plt.show()

print(f"Broj boja u kvantiziranoj slici: {len(np.unique(img_array_aprox, axis = 0))}")

img_array_aprox = img_array.copy()

J_values = []
for i in range(1,15):
    km = KMeans(n_clusters = i, init="k-means++", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    J_values.append(km.inertia_)

plt.figure()
plt.plot(range(1,15), J_values, marker=".")
plt.title("Lakat metoda")
plt.xlabel("K")
plt.ylabel("J")
plt.tight_layout()
plt.show()

unique_labels = np.unique(labels)
for i in range (len(unique_labels)):
    binary_image = labels == unique_labels[i]
    binary_image = np.reshape(binary_image, (w,h))
    plt.figure()
    plt.title(f"Binarna slika {i+1}. grupe boja")
    plt.imshow(binary_image)
    plt.tight_layout()
    plt.show()