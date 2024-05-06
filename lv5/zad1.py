import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="viridis", marker="o", label="trening podaci")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="hot", marker="x", label="test podaci")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

LogRegression_model = LogisticRegression ()
LogRegression_model.fit( X_train , y_train )

b = LogRegression_model.intercept_[0]
w1, w2 = LogRegression_model.coef_.T

print(f"b: {b}\n w1,w2: {w1}, {w2}")

plot_decision_regions(X_train, y_train, LogRegression_model)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap="viridis")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

y_test_p = LogRegression_model.predict( X_test )

cm = confusion_matrix ( y_test , y_test_p )
print (" Matrica zabune : " , cm )
disp = ConfusionMatrixDisplay ( confusion_matrix ( y_test , y_test_p ) )
disp . plot ()
plt . show ()
print(f"Tocnost: {accuracy_score(y_test, y_test_p)}")
print(classification_report(y_test, y_test_p))

colors = np.where(((y_test == 1) & (y_test_p == 1)) | ((y_test == 0) & (y_test_p == 0)), 'green', 'black')

plt.scatter(X_test[:,0], X_test[:,1], c=colors)
plt.show()