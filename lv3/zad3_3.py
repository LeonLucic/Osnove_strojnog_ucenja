import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Učitavanje Iris dataseta
iris = datasets.load_iris()
X = iris.data  # podaci o latama i čašicama
y = iris.target  # oznake klasa

# Podjela podataka na skup za učenje i skup za testiranje u omjeru 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Skaliranje ulaznih podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konvertiranje izlaznih podataka u kategorikalne veličine
y_train_categorical = np.eye(3)[y_train]
y_test_categorical = np.eye(3)[y_test]

# a) Izgradnja neuronske mreže
model = Sequential([
    Dense(12, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(7, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='relu'),
    Dense(3, activation='softmax')
])

# Ispis informacija o mreži
model.summary()

# b) Podešavanje procesa treniranja
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=[Accuracy()])

# c) Treniranje mreže
history = model.fit(X_train_scaled, y_train_categorical, epochs=450, batch_size=7, validation_split=0.1)

# d) Pohrana modela na disk
model.save('iris_model.h5')

# e) Evaluacija mreže na testnom skupu podataka
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_categorical)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# f) Predikcija mreže na testnom skupu podataka i prikaz matrice zabune
y_pred = np.argmax(model.predict(X_test_scaled), axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)

# Prikaz matrice zabune
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Komentar: Matrica zabune prikazuje koliko je model pogodio svaku od klasa na testnom skupu podataka.
#            Možemo vidjeti kako se model ponaša u predviđanju različitih klasa.
#            Ako je potrebno poboljšati rezultate, možemo pokušati s različitim arhitekturama neuronske mreže,
#            optimizirati hiperparametre ili dodatno obraditi ulazne podatke. Također, mogli bismo eksperimentirati
#            s dodatnim tehnologijama poput tehnika izbjegavanja prenaučenosti (npr. regularizacija) ili
#            optimizacije hiperparametara (npr. križna validacija).
