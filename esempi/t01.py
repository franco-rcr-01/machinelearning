# Classificazione con il classificatore di k-nearest neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il classificatore KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

# addestra il classificatore con il dataset iris
knn.fit(iris.data, iris.target)

# effettua una predizione su un nuovo campione
new_sample = [[5.0, 3.6, 1.3, 0.25]]
predicted_class = knn.predict(new_sample)
print("Classe predetta:", predicted_class)

new_sample = [[2.0, 1.6, 5.3, 1.25]]
predicted_class = knn.predict(new_sample)
print("Classe predetta:", predicted_class)

