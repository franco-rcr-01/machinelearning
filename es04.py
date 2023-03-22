import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np  
import pandas as pd
import sys

# Assign colum names to the dataset
names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Read dataset to pandas dataframe
dataset = pd.read_csv("diabetes.csv", names=names)  

print("Intestazione (e prime 4 righe)\n{0}".format(dataset.head(4)))
print("I primi 4 valori\n{0}".format(dataset.iloc[:4].values))

# Training e testing...
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values  
sys.exit(-1)
#Il metodo di splitting
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# #preprocessing e standardizzazione
# from sklearn.preprocessing import StandardScaler  

# scaler = StandardScaler()  
# scaler.fit(X_train)

# print(X_train)
# X_train = scaler.transform(X_train)  
# X_test = scaler.transform(X_test)  
# print(X_train)


#Applica il classificatore
from sklearn.neighbors import KNeighborsClassifier 
# classifier = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree")
# classifier = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
# classifier = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
# classifier = KNeighborsClassifier(n_neighbors=5, algorithm="auto")
classifier.fit(X_train, y_train)  

#Predizioni e valutazione della qualità
y_pred = classifier.predict(X_test)  

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))



# #Valutazione del valore ottimale per K
# error = []

# # Calculating error for K values between 1 and 20
# for i in range(1, 20):  
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error.append(np.mean(pred_i != y_test))

# plt.figure(figsize=(12, 6))  
# plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
#          markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')  
# plt.xlabel('K Value')  
# plt.ylabel('Mean Error')  
# plt.show()
