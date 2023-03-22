from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il modello K-means con k=3
kmeans = KMeans(n_clusters=150, n_init="auto")

# addestra il modello con il dataset iris
kmeans.fit(iris.data)

# effettua una predizione sui campioni del dataset
predicted_labels = kmeans.predict(iris.data)

