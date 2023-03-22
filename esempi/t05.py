# Random Forest per la classificazione

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il classificatore Random Forest con 100 alberi
rf = RandomForestClassifier(n_estimators=100)

# addestra il classificatore con il dataset iris
rf.fit(iris.data, iris.target)

# effettua una predizione su un nuovo campione
new_sample = [[5.0, 3.6, 1.3, 0.25]]
predicted_class = rf.predict(new_sample)

print("Classe predetta:", predicted_class)
