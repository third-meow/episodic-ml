import numpy as np
import random
import pydotplus as pdp
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

print('\n\n')

iris = load_iris()
train_data, test_data, train_labels, test_labels = train_test_split(iris.data,
        iris.target, test_size = 0.3)


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
prediction = clf.predict(test_data)

print(accuracy_score(test_labels, prediction))

