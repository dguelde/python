import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
X_train = iris.data[iris.target != 2, :2]  # we only take the first two features.
y_train = iris.target[iris.target != 2]

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
colors = ["red" if yi==1 else "blue" for yi in y_train]
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max);