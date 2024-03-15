import numpy as np
import matplotlib . pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", dtype="str")
data = data[1::]
data = np.array(data, np.float64)
print(f"People measured: {len(data)}")
height = data[:,1]
weight = data[:,2]
plt.scatter(height, weight, linewidth = 0.1)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

min = height.min()
max = height.max()
mean = height.mean()

height = data[0::50, 1]
weight = data[0::50, 2]
plt.scatter(height, weight, linewidth = 0.1)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

print(f"Min: {min} Max: {max} Mean: {round(mean,2)} ")
men = data[data[:,0] == 1]
women = data[data[:,0] == 0]
print(f"Men: Min: {men[:,1].min()} Max: {men[:,1].max()} Mean: {round(men[:,1].mean(),2)} ")
print(f"Women: Min: {women[:,1].min()} Max: {women[:,1].max()} Mean: {round(women[:,1].mean(),2)} ")