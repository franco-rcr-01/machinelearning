from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.data[iris.target==1][:5])
print(iris.data[iris.target==1, 0][:5])

print(iris["target_names"])
print(iris.target_names)
n_samples, n_features = iris.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
# the sepal length, sepal width, petal length and petal width of the first sample (first flower)
print(iris.data[0])
print(iris.data.shape)
print(iris.target.shape)
print(iris.target)
print(iris.target_names)


for x_index in range(4):
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green']

    for label, color in zip(range(len(iris.target_names)), colors):
        ax.hist(iris.data[iris.target==label, x_index], 
                label=iris.target_names[label],
                color=color)

    ax.set_xlabel(iris.feature_names[x_index])
    ax.legend(loc='upper right')
    plt.show()
