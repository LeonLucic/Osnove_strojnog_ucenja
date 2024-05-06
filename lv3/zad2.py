import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

#Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.

plt.hist(data['CO2 Emissions (g/km)'], bins=30, color="lightblue")
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.show()

#Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
#C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
#velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva

print(data['Fuel Type'].unique())

colors = ['green', 'red', 'blue', 'yellow']
for index, fuel_type in zip(range(len(data['Fuel Type'].unique())), data['Fuel Type'].unique()):
    data_fuel_type = data[data['Fuel Type'] == fuel_type]
    plt.scatter(data_fuel_type['Fuel Consumption City (L/100km)'], data_fuel_type['CO2 Emissions (g/km)'], label=fuel_type, color=colors[index])

plt.legend()
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()


#Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
#goriva. Primjecujete li grubu mjernu pogrešku u podacima?

data.boxplot(column="Fuel Consumption Hwy (L/100km)", by="Fuel Type")
plt.xlabel('Fuel Type')
plt.ylabel('Fuel Consumption Hwy (L/100km)')
plt.show()

#Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
#groupby.

data_groupedby_fuel_type = data.groupby('Fuel Type')['Make'].agg('count')
plt.bar(data_groupedby_fuel_type.index, data_groupedby_fuel_type.values)
plt.xlabel('Fuel Type')
plt.ylabel('Number of Vehicles')
plt.show()

#Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
#broj cilindara. 

plt.bar(data['Cylinders'], data['CO2 Emissions (g/km)'])
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()