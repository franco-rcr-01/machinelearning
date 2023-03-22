# Random Forest per la classificazione

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# carica il dataset iris
iris = load_iris()

# inizializza il classificatore Random Forest con 100 alberi
rf = RandomForestClassifier(n_estimators=10)

# addestra il classificatore con il dataset iris
rf.fit(iris.data, iris.target)

# effettua una predizione su un nuovo campione
for i in range(150):
    print("Classe predetta({}):{}".format(iris.target[i], rf.predict([iris.data[i]])))
