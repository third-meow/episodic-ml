import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()

test_index = [0,50,100]

training_labels = np.delete(iris.target, test_index)
training_data = np.delete(iris.data, test_index, axis = 0)

test_labels = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_labels)

