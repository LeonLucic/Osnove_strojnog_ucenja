import pandas as pd
import numpy as np


# a)
data = pd.read_csv('data_C02_emission.csv')

print(data.info())
print(len(data))

print("Null: " + str(data.isnull().sum()))
print("Duplicated: " + str(data.duplicated().sum()))

data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')


# b)

sorted_data = data.sort_values(by='Fuel Consumption City (L/100km)')
print("Najmanja potrosnja:")
print(sorted_data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3))
print("Najveca potrosnja:")
print(sorted_data[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3))


# c)

filtered_data = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(len(filtered_data))
print(filtered_data['CO2 Emissions (g/km)'].mean())

# d)
print("Audi:")
filtered_data = data[data['Make'] == "Audi"]
print(len(filtered_data))
filtered_data = filtered_data[filtered_data['Cylinders'] == 4]
print(filtered_data['CO2 Emissions (g/km)'].mean())

# e)
print("4 cilindra")
filtered_data = data[data['Cylinders'] == 4]
print(len(filtered_data))
print(filtered_data['CO2 Emissions (g/km)'].mean())

print("6 cilindra")
filtered_data = data[data['Cylinders'] == 6]
print(len(filtered_data))
print(filtered_data['CO2 Emissions (g/km)'].mean())

print("8 cilindra")
filtered_data = data[data['Cylinders'] == 8]
print(len(filtered_data))
print(filtered_data['CO2 Emissions (g/km)'].mean())

# f)

print("Diesel: ")
filtered_data = data[data['Fuel Type'] == "D"]
print(filtered_data["Fuel Consumption City (L/100km)"].mean())
print(filtered_data["Fuel Consumption City (L/100km)"].median())

print("Regularni benzin: ")
filtered_data = data[data['Fuel Type'] == "X"]
print(filtered_data["Fuel Consumption City (L/100km)"].mean())
print(filtered_data["Fuel Consumption City (L/100km)"].median())

# g)
print("Najveci potrosac:")
filtered_data = data[(data['Fuel Type'] == "D") & (data['Cylinders'] == 4)]
filtered_data = filtered_data.sort_values(by="Fuel Consumption City (L/100km)")
print(filtered_data.head(1))

# h)
print("Manualci:")
filtered_data = data[data['Transmission'].str.contains('M')]
print(len(filtered_data))

# i)
print("Korelacija:")
print(data.corr(numeric_only=True))



