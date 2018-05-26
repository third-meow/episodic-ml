import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydotplus as pdp

iris = load_iris()

test_index = [0,50,100]

training_labels = np.delete(iris.target, test_index)
training_data = np.delete(iris.data, test_index, axis = 0)

test_labels = iris.target[test_index]
test_data = iris.data[test_index]

clf = tree.DecisionTreeClassifier()
clf.fit(training_data, training_labels)

print(test_labels)
print(clf.predict(test_data))


graph_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,filled=True,
                                    rounded=True,special_characters=True)


graph = pdp.graph_from_dot_data(graph_data)
graph.write_pdf('irisTree.pdf')
