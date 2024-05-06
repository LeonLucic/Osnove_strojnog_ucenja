import numpy as np
import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
#duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
#category

print(f"Broj mjerenja: {len(data)}")
print(data.dtypes)
data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']] = data[['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']].astype('category')
print(f"Broj izostalih vrijednosti: {data.isnull().sum()}")
print(f"Broj izostalih vrijednosti: {data.duplicated().sum()}")


#Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
#ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja.

cars_sortedby_city_consumption = data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False)
print(f"Automobili s najvecom gradskom potrosnjom: \n{cars_sortedby_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].head(3)}")
print(f"Automobili s najmanjom gradskom potrosnjom: \n{cars_sortedby_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']].tail(3)}")

#Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
#plinova za ova vozila?

cars_with_selected_engine_size = data[data['Engine Size (L)'].between(2.5, 3.5)]
print(f"Broj vozila s velicinom motora izmedju 2.5 i 3.5 L: {len(cars_with_selected_engine_size)}")
print(f"Prosjecna emisija plinova za vozila s velicinom motora izmedju 2.5 i 3.5 L: {cars_with_selected_engine_size['CO2 Emissions (g/km)'].mean()}")

#Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
#plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara?

audi_cars = data[data['Make'] == 'Audi']
print(f"Broj Audi automobila: {len(audi_cars)}")
audi_cars_with_four_cylinders = audi_cars[audi_cars['Cylinders'] == 4]
print(f"Prosjecna emisija CO2 plinova audi automobila koji imaju 4 cilindra: {audi_cars_with_four_cylinders['CO2 Emissions (g/km)'].mean().__round__(2)}")

#Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
#broj cilindara?

data_groupedby_cylinders = data.groupby('Cylinders')
print(f"Broj vozila prema broju cilindara: {data_groupedby_cylinders['Cylinders'].count()}")
print(f"Prosjecna emisija CO2 plinova prema broju cilindara: {data_groupedby_cylinders['CO2 Emissions (g/km)'].mean().__round__(2)}")

#Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
#koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?

dizel_cars = data[data['Fuel Type'] == 'D']
benzin_cars = data[data['Fuel Type'] == 'E']

print(f"Prosjecna gradska potrosnja dizelasa: {dizel_cars['Fuel Consumption City (L/100km)'].mean().__round__(2)}")
print(f"Prosjecna gradska potrosnja benzinaca: {benzin_cars['Fuel Consumption City (L/100km)'].mean().__round__(2)}")

print(f"Medijalne vrijednosti gradske potrosnje goriva za dizelase: {dizel_cars['Fuel Consumption City (L/100km)'].median()}")
print(f"Medijalne vrijednosti gradske potrosnje goriva za benzince: {benzin_cars['Fuel Consumption City (L/100km)'].median()}")

#Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?

cars_with_diesel_and_four_cylinders = dizel_cars[dizel_cars['Cylinders'] == 4]
car_with_diesel_and_four_cylinders_with_max_city_consumption = cars_with_diesel_and_four_cylinders.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1)
print(f"Dizelas s 4 cilindra s najvecom gradskom potrosnjom goriva: \n{car_with_diesel_and_four_cylinders_with_max_city_consumption}")

#Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)?

manual_cars = data[data['Transmission'].str.startswith('M')]
print(f"Broj vozila s rucnim mjenjacem: {len(manual_cars)}")

# Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat.

print(data.corr(numeric_only = True ))