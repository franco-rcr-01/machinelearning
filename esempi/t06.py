# Principal Component Analysis (PCA) per la riduzione della dimensionalità

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il modello PCA con due componenti principali
pca = PCA(n_components=2)

# trasforma i dati in due dimensioni
transformed_data = pca.fit_transform(iris.data)

# visualizza i dati trasformati
import matplotlib.pyplot as plt

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=iris.target)
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.show()
