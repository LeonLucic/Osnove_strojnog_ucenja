from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# Učitavanje Iris dataseta
iris = datasets.load_iris()
X = iris.data  # podaci o latama i čašicama
y = iris.target  # oznake klasa
target_names = iris.target_names

# a) Prikažite odnos duljine latice i čašice svih pripadnika klase Virginica pomoću scatter dijagrama zelenom bojom.
#    Dodajte na isti dijagram odnos duljine latice i čašice svih pripadnika klase Setosa, sivom bojom.
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 2, 0], X[y == 2, 1], c='green', label='Virginica')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='gray', label='Setosa')
plt.xlabel('Duljina latica (cm)')
plt.ylabel('Duljina čašice (cm)')
plt.title('Odnos duljine latice i čašice za klase Virginica i Setosa')
plt.legend()
plt.show()
# Komentar: Scatter dijagram prikazuje odnos duljine latice i čašice za klase Virginica i Setosa.
#            Možemo primijetiti da postoji jasna razlika između ove dvije klase u odnosu duljine latice i čašice.

# b) Pomoću stupčastog dijagrama prikažite najveću vrijednost širine čašice za sve tri klase cvijeta.
plt.figure(figsize=(8, 6))
max_sepal_width = [np.max(X[y == i, 1]) for i in range(3)]
plt.bar(target_names, max_sepal_width, color=['blue', 'orange', 'green'])
plt.xlabel('Klasa cvijeta')
plt.ylabel('Najveća širina čašice (cm)')
plt.title('Najveća vrijednost širine čašice za svaku klasu cvijeta')
plt.show()
# Komentar: Stupčasti dijagram jasno prikazuje najveću vrijednost širine čašice za svaku od tri klase cvijeta.

# c) Koliko jedinki pripadnika klase Setosa ima veću širinu čašice od prosječne širine čašice te klase?
setosa_sepal_width = X[y == 0, 1]  # Širina čašice za klase Setosa
avg_setosa_sepal_width = np.mean(setosa_sepal_width)
count_above_avg = np.sum(setosa_sepal_width > avg_setosa_sepal_width)
print("Broj jedinki klase Setosa s većom širinom čašice od prosjeka: ", count_above_avg)
