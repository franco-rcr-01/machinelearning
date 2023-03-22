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
for i in range(150):
    print("Classe predetta({}):{}".format(iris.target[i], svm.predict([iris.data[i]])))

# # predicted_labels = list(map( lambda x : iris.target[x], predicted_labels))

# print("Etichette predette:", predicted_class)
# print("Etichette del set :", iris.target)

# from sklearn.metrics import classification_report, confusion_matrix  

# print(confusion_matrix(iris.target, predicted_class))
# # print(classification_report(iris.target, predicted_labels))
