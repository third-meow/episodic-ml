import pandas as pd
from sklearn import tree

pd.set_option('display.max_rows', 1000)

def main():
    df = pd.read_csv('data/train.csv')
    survived = df['Survived']

    # not using all colunms
    df = df.drop(['Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    #replace any missing ages with the mean age
    df = df.fillna(round(df['Age'].mean()))

    #replace strings with floats
    df = df.replace('male', 0.0)
    df = df.replace('female', 1.0)

    split = int(df.size*4/5)
    test_data = df.iloc[:split]
    test_labels = survived.iloc[:split]

    train_data = df.iloc[split:]
    train_labels = survived.iloc[split:]
    

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_labels)

    tree.plot_tree(clf)






if __name__ == '__main__':
    main()
