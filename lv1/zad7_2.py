""""Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 70:30.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import MinMaxScaler
from sklearn . neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("titanic.csv")

#izbacivanje null i izostalih vrijednosti
data.dropna(inplace=True)

#podjela na test i train
X=data[['Pclass', 'Sex', 'Fare','Embarked']]
y=data['Survived']

#pretvaranje kategorickih varijabli u numericke
X=pd.get_dummies(X, columns=["Sex", "Embarked"])

#podjela 70:30
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.3, random_state =1)

#skaliranje
sc=MinMaxScaler()
X_train_n=sc.fit_transform( X_train )
X_test_n=sc.transform( X_test )

# a)Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Vizualizirajte podatkovne
# primjere i granicu odluke.

# izrada KNN 
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

KNN_model = KNeighborsClassifier( n_neighbors = 5 )
KNN_model.fit( X_train_n , y_train )
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict( X_test_n )

# b) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje.

print("KNN(neighbours = 5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#Sto je k manji model je slozeniji, a sto je k veci model postaje prejednostavan. K=n prejednostavno

#c) Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma
# KNN.


KNN2_model = KNeighborsClassifier(n_neighbors=3)
param_grid = {"n_neighbors": np.arange(1,100)}
KNN2_gscv = GridSearchCV(KNN2_model, param_grid, cv=5, scoring='accuracy')
KNN2_gscv.fit(X_train_n, y_train)
print(KNN2_gscv.best_params_)
print(KNN2_gscv.best_score_)

# d) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje
# za dobiveni K. Usporedite dobivene rezultate s rezultatima kada je K=5.

y_train_knn2 = KNN2_gscv.predict(X_train_n)
y_test_knn2 = KNN2_gscv.predict(X_test_n)

print("KNN grid search: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_knn2))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_knn2))))

scores_5 = cross_val_score(KNN_model, X_train_n, y_train, cv=5)
print(scores_5)

scores_3 = cross_val_score(KNN2_model, X_train_n, y_train, cv=5)
print(scores_3)


 #Evo ispravljenog koda za SVM:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Učitavanje podataka
data = pd.read_csv("titanic.csv")

# Izbacivanje nedostajućih vrijednosti
data.dropna(inplace=True)

# Podjela na ulazne i izlazne podatke
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
y = data['Survived']

# Podjela na skup za učenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Skaliranje podataka
sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Kreiranje SVM modela
svm_model = SVC()

# Treniranje modela
svm_model.fit(X_train_scaled, y_train)

# Predikcija na skupu za učenje i testiranje
y_train_pred = svm_model.predict(X_train_scaled)
y_test_pred = svm_model.predict(X_test_scaled)

# Izračun točnosti
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Točnost na skupu za učenje: {:.2f}".format(train_accuracy))
print("Točnost na skupu za testiranje: {:.2f}".format(test_accuracy))

# Unakrsna validacija za određivanje optimalnih hiperparametara
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
svm_grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)

print("Najbolji hiperparametri:", svm_grid.best_params_)
print("Najbolja točnost:", svm_grid.best_score_)

# Evaluacija modela s najboljim hiperparametrima
best_svm_model = svm_grid.best_estimator_
y_train_pred_best = best_svm_model.predict(X_train_scaled)
y_test_pred_best = best_svm_model.predict(X_test_scaled)

train_accuracy_best = accuracy_score(y_train, y_train_pred_best)
test_accuracy_best = accuracy_score(y_test, y_test_pred_best)

print("Točnost na skupu za učenje (najbolji model): {:.2f}".format(train_accuracy_best))
print("Točnost na skupu za testiranje (najbolji model): {:.2f}".format(test_accuracy_best))

# Unakrsna validacija s najboljim hiperparametrima
cross_val_scores = cross_val_score(best_svm_model, X_train_scaled, y_train, cv=5)
print("Unakrsna validacija (najbolji model):", cross_val_scores)


#a korištenje algoritma K-Means:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Učitavanje podataka
data = pd.read_csv("titanic.csv")

# Izbacivanje nedostajućih vrijednosti
data.dropna(inplace=True)

# Podjela na ulazne i izlazne podatke
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
X = pd.get_dummies(X, columns=["Sex", "Embarked"])
y = data['Survived']

# Podjela na skup za učenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Skaliranje podataka
sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Kreiranje K-Means modela
kmeans = KMeans(n_clusters=2, random_state=1)

# Treniranje modela
kmeans.fit(X_train_scaled)

# Predikcija na skupu za testiranje
y_test_pred = kmeans.predict(X_test_scaled)

# Prilagodba predikcija da odgovaraju formatu stvarnih vrijednosti
y_test_pred_adjusted = np.abs(y_test_pred - 1)

# Izračun točnosti
test_accuracy = accuracy_score(y_test, y_test_pred_adjusted)

print("Točnost na skupu za testiranje: {:.2f}".format(test_accuracy))
