import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', skiprows=1, delimiter=',')
print(f"Mjerenja su izvrsena na: {data[:,0].size} ljudi")

visina = data[:,1]
masa = data[:,2]
plt.scatter(visina, masa)
plt.xlabel('visina')
plt.ylabel('masa')
plt.title('Odnos visine i mase')
plt.show()

visina_svakih_pedeset = data[::50, 1]
masa_svakih_pedeset = data[::50, 2]
plt.scatter(visina_svakih_pedeset, masa_svakih_pedeset)
plt.xlabel('visina')
plt.ylabel('masa')
plt.title('Odnos visine i mase')
plt.show()

print(f"Minimalna vrijednost visine osobe: {visina.min()}")
print(f"Maksimalna vrijednost visine osobe: {visina.max()}")
print(f"Srednja vrijednost visine osobe: {visina.mean()}")

visina_muskaraca = data[data[:,0] == 1, 1]
visina_zena = data[data[:,0] == 0, 1]

print(f"Minimalna vrijednost visine muskaraca: {visina_muskaraca.min()}")
print(f"Maksimalna vrijednost visine muskaraca: {visina_muskaraca.max()}")
print(f"Srednja vrijednost visine muskaraca: {visina_muskaraca.mean()}")

print(f"Minimalna vrijednost visine zena: {visina_zena.min()}")
print(f"Maksimalna vrijednost visine zena: {visina_zena.max()}")
print(f"Srednja vrijednost visine zena: {visina_zena.mean()}")