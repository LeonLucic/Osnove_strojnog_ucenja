"""Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 75:25.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:
a) Izgradite neuronsku mrežu sa sljede´cim karakteristikama:
- model oˇcekuje ulazne podatke X
- prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
- drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
- tre´ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
- izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
Ispišite informacije o mreži u terminal.
b) Podesite proces treniranja mreže sa sljede´cim parametrima:
- loss argument: binary_crossentropy
- optimizer: adam
- metrika: accuracy.
c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i veliˇcinom
batch-a 5.
d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
e) Izvršite evaluaciju mreže na testnom skupu podataka.
f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
ako je potrebno."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn . model_selection import cross_val_score

from tensorflow import keras
from keras import layers
from keras.models import load_model


#učitavanje dataseta
data = pd.read_csv('priprema_LV2/data/titanic.csv')

#izbacivanje null i izostalih vrijednosti
data = data.drop_duplicates()
data = data.dropna(axis=0)
data.reset_index()

#pretvaranje kategorickih varijabli u numericke
data = pd.get_dummies(data, columns = ['Sex', 'Embarked'], dtype=float)

X = data[['Pclass', 'Sex_male','Sex_female', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# train test split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.25, random_state=1)

sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )

#a)
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(7,)))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

#b)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy",]
)

# c)ucenje mreze
history = model.fit(
    X_train_n, 
    y_train,
    batch_size = 5,
    epochs = 100,
    validation_split = 0.1
    )

# d)
model.save("priprema_LV2/FCN/model.keras")

# e)evaluacija moreže
del model
model = load_model("priprema_LV2/FCN/model.keras")

test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
print(test_acc)
print(test_loss)

# f)predikciju mreže,matrica zabune
y_test_p = model.predict(X_test_n)
y_test_p = (y_test_p > 0.5).astype(int)
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay ( cm )
disp.plot()
plt.show()

print(classification_report(y_test, y_test_p))
