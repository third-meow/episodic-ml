import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib

pd.set_option('display.max_rows', 1000)

def main():
    df = pd.read_csv('data/train.csv')
    survived = df['Survived']

    # not using all columns
    df = df.drop(['Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    # replace any missing ages with the mean age
    df = df.fillna(round(df['Age'].mean()))

    # replace strings with floats
    df = df.replace('male', 0.0)
    df = df.replace('female', 1.0)


    # split data and labels into training and test sets
    split_msk = np.random.rand(len(df)) < 0.8

    train_data = df[split_msk]
    train_labels = survived[split_msk]

    test_data = df[~split_msk]
    test_labels = survived[~split_msk]


    # create decision tree
    clf = tree.DecisionTreeClassifier()
    # train decision tree
    clf = clf.fit(train_data, train_labels)

    # calculate accuracy
    test_pred = clf.predict(test_data)

    print_err_mat(test_pred, test_labels)


if __name__ == '__main__':
    main()
