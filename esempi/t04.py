# Support Vector Machine (SVM) per la classificazione:

from sklearn.svm import SVC
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il classificatore SVM con kernel lineare
svm = SVC(kernel='linear')

# addestra il classificatore con il dataset iris
svm.fit(iris.data, iris.target)

# effettua una predizione su un nuovo campione
new_sample = [[5.0, 3.6, 1.3, 0.25]]
predicted_class = svm.predict(new_sample)

print("Classe predetta:", predicted_class)
