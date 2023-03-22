import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np  
import pandas as pd

import sys

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

print("Intestazione (e prime 4 righe)\n{0}".format(dataset.head(4)))
print("I primi 4 valori\n{0}".format(dataset.iloc[:4].values))

# Training e testing...
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values  

#Il metodo di splitting
from sklearn.model_selection import train_test_split

datasets = train_test_split(X, y, test_size=0.20)  
train_data, test_data, train_labels, test_labels = datasets

from sklearn.linear_model import Perceptron

p = Perceptron(random_state=42,
               max_iter=50,
               tol=0.001)
p.fit(train_data, train_labels)

import random

sample = random.sample(range(len(train_data)), 10)
for i in sample:
    print(i, p.predict([train_data[i]]))


from sklearn.metrics import classification_report
print(classification_report(p.predict(train_data), train_labels))

print(classification_report(p.predict(test_data), test_labels))
# https://meet.google.com/uoy-fbbr-ptk
