from sklearn import datasets
from sklearn . model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn . preprocessing import MinMaxScaler

data = pd.read_csv('data_C02_emission.csv')
 
#a)
numerical_features = ['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)', 
                      'Engine Size (L)', 'Cylinders']
x = data[numerical_features].to_numpy()
y = data['CO2 Emissions (g/km)'].to_numpy()  #zadatak kaze model koji procjenjuje emisiju CO2 plinova..zato je to izlazni
 
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2, random_state = 1) #dijelimo ih 80-20


#b) za svaku numeričku velicinu poseban scatter ovisnosti s emisijom co2 plinova
plt.scatter(x_train[:, 0], y_train, color="green", label="trening")
plt.scatter(x_test[:, 0], y_test, color="yellow", label="test")
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()
 
plt.scatter(x_train[:,1], y_train, color="green")
plt.scatter(x_test[:,1], y_test, color="yellow")
plt.xlabel('Fuel Consumption Hwy (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

 
#c) standardizacija ulaznih veličina skupa za učenje
scaler = MinMaxScaler()
plt.figure()
plt.hist(x_train[:, 0], bins=20, color="green", edgecolor='black')  #prikaz jedne ulazne velicine prije skaliranja
plt.show()
x_train_n = scaler.fit_transform(x_train)
x_test_n = scaler.transform(x_test)
 
plt.figure()
plt.hist(x_train_n[:, 0], bins=20, color="yellow",  edgecolor='black')  #prikaz jedne ulazne velicine nakon skaliranja
plt.show()
 
 
#d) linearni regresijski model
linearModel = lm.LinearRegression()
linearModel.fit(x_train_n, y_train)
print(linearModel.coef_)
 
 
# e) Procjena izlazne veličine, dijagrama raspršenja - odnos između stvarnih vrijednosti izlazne velicine i procjene dobivene modelom
y_prediction = linearModel.predict(x_test_n)   
plt.figure()
plt.ylabel('Predicted')
plt.xlabel('Actual')
print("Shape of y_test:", y_test.shape)

plt.scatter(x=y_test, y=y_prediction)
plt.show()
 
# f) Vrednovanje - vrijednost regresijskih metrika
print('MSE - Mean squared error: {:.5f}'.format(mean_squared_error(y_test, y_prediction)))
print('MSE - Mean squared error: {:.5f}'.format(np.sqrt(mean_squared_error(y_test, y_prediction))))
print('MAE - Mean absolute error: {:.5f}'.format(mean_absolute_error(y_test, y_prediction)))
print('MAPE - Mean absolute percentage error: {:.5f}'.format(mean_absolute_percentage_error(y_test, y_prediction))+'%')
print('R^2: {:.5f}'.format(r2_score(y_test, y_prediction)))

