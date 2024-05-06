import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
 
data = pd.read_csv('data_C02_emission.csv')
 
#  1-od-K kodiranje kategorickih velicina
encoder = OneHotEncoder()
encoder_df = pd.DataFrame(encoder.fit_transform(data[['Fuel Type']]).toarray()) #dodali smo jos  jednu kategoricku velicinu
data = data.join(encoder_df)
 
data.columns = ['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']
y = data['CO2 Emissions (g/km)'].copy()
X = data.drop('CO2 Emissions (g/km)', axis=1)
X_train_all , X_test_all , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)
 
X_train = X_train_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
X_test = X_test_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
 
linearModel = lm.LinearRegression()
linearModel.fit(X_train,y_train)
y_prediction = linearModel.predict(X_test)
 
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_test, c='b',label='Real values', s=5, alpha=0.5)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_prediction, c='r',label='Prediction', s=5, alpha=0.5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()
 
#maksimalna pogreska u procjeni 
maxError = max_error(y_test,y_prediction)
print('Max pogreška u procjeni: {:.3f}'.format(maxError))
print(f"Model vozila s max pogreškom u procjeni: {X_test_all[abs(y_test-y_prediction) == maxError]['Model'].iloc[0]}")
