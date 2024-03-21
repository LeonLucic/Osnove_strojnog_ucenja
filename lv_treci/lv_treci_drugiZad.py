import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

# pod a)
plt.figure()
data['Fuel Consumption City (L/100km)'].plot(kind ='hist', bins=20)

# pod b)
data.plot.scatter(x='Fuel Consumption City (L/100km)',
                  y='CO2 Emissions (g/km)',
                  c='Fuel Type',
                  cmap ="nipy_spectral")

# pod c)
data.plot.box(column='Fuel Consumption Hwy (L/100km)', by="Fuel Type")

# pod d)
plt.figure()
plt.title("Counts")
grouped_data = data.groupby("Fuel Type")['Fuel Type'].count()
grouped_data.plot.bar()

# pod e)
plt.figure()
plt.title("CO2 prema broju cilindara")
grouped_data = data.groupby("Cylinders")['CO2 Emissions (g/km)'].mean()
grouped_data.plot.bar()


plt.show()
