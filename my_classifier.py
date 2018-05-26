import numpy as np
import random
import pydotplus as pdp
from scipy.spatial import distance
#from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#returns euclididean distance between a and b
def euc(a,b):
    return distance.euclidean(a,b)

#the classifier - based on k-nearest-neighbors
class myClassifier():
    #stores training data
    def fit(self, trn_data, trn_labels):
        self.train_data = trn_data
        self.train_labels = trn_labels

    #predict label based on some data
    def predict(self, tst_data):
        self.prediction = []
        #for all data points in the input data
        for point in tst_data:
            #label found by self.closest function
            label = self.closest(point)
            #add label to list of prediction output
            self.prediction.append(label)
        return self.prediction
    
    #returns label of closest neighbour
    def closest(self, pnt):
        best_dist = euc(pnt, self.train_data[0])
        best_index = 1
        #look through all training data
        for i in range(1, len(self.train_data)):
            #mesure distance between point in question and current training data
            #point
            dist = euc(pnt, self.train_data[i])
            #if distence is smaller than the smallest so far, set smallest so
            #far to current distance and store current index as best index
            if dist < best_dist:
                best_dist = dist
                best_index = i
        #return label of closest training point
        return self.train_labels[best_index]


print('\n\n')

iris = load_iris()
train_data, test_data, train_labels, test_labels = train_test_split(iris.data,
        iris.target, test_size = 0.3)


clf = myClassifier()
clf.fit(train_data, train_labels)
prediction = clf.predict(test_data)

print(accuracy_score(test_labels, prediction))

