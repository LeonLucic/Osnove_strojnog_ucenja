from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Učitavanje Iris dataseta
iris = datasets.load_iris()
X = iris.data  # podaci o latama i čašicama
y = iris.target  # oznake klasa

# a) Pronalaženje optimalnog broja klastera K pomoću metode lakta
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# b) Grafički prikaz lakta metode
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Inercija')
plt.title('Metoda lakta za pronalaženje optimalnog K')
plt.show()

# c) Primjena algoritma K-srednjih vrijednosti
optimal_k = 3  # prema metodi lakta
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)

# d) Dijagram raspršenja klasterskih grupa
plt.figure(figsize=(8, 6))

# Prikazivanje svih podataka
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50, label='Stvarne klase')

# Prikazivanje centroida
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroidi')

plt.xlabel('Dužina latice (cm)')
plt.ylabel('Širina latice (cm)')
plt.title('K-srednje vrijednosti za Iris Dataset')
plt.legend()
plt.show()

# e) Usporedba dobivenih klasa sa stvarnim vrijednostima
# Pogledamo koliko se klasteri podudaraju sa stvarnim klasama
# Zamijenimo oznake klastera sa stvarnim klasama
cluster_labels = np.zeros_like(kmeans.labels_)
for i in range(3):
    mask = (kmeans.labels_ == i)
    cluster_labels[mask] = np.bincount(y[mask]).argmax()

# Izračunajmo točnost klasifikacije
accuracy = np.mean(cluster_labels == y)
print("Točnost klasifikacije: {:.2f}%".format(accuracy * 100))
