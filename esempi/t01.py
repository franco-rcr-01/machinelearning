# Classificazione con il classificatore di k-nearest neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il classificatore KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)

# addestra il classificatore con il dataset iris
knn.fit(iris.data, iris.target)

for i in range(3):
    print(iris.data[i*50])
    print(iris.target[i*50])

# effettua una predizione su un nuovo campione
new_sample = [[5.0, 3.6, 1.3, 0.25]]
predicted_class = knn.predict(new_sample)
print("Classe predetta:", predicted_class)

new_sample = [[2.0, 1.6, 5.3, 1.25]]
predicted_class = knn.predict(new_sample)
print("Classe predetta:", predicted_class)

ns=[[5.1, 3.5, 1.4, 0.2]]
predicted_class = knn.predict(ns)
print("Classe predetta (0):", predicted_class)

ns=[[7.,  3.2 ,4.7, 1.4]]
predicted_class = knn.predict(ns)
print("Classe predetta (1):", predicted_class)

ns=[[6.3, 3.3, 6.,  2.5]]
predicted_class = knn.predict(ns)
print("Classe predetta (2):", predicted_class)

for i in range(iris.data.size):
    ns=[iris.data[i]]
    print("Classe predetta ({}): {}".format(iris.target[i], knn.predict(ns)))