from sklearn import tree

#[height,weight, shoe-size]
training_entity_values = [[180, 90, 44], [170, 60, 44], [200, 100, 46], [170, 90, 35], [180, 80, 44], [160, 50, 36], [175, 60, 39], [194, 80, 41]]

#male/female
training_entity_sex = ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(training_entity_values, training_entity_sex)

n_height = int(input('enter height of new person: '))
n_weight = int(input('enter weight of new person: '))
n_shoesize = int(input('enter shoe size of new person: '))

prediction = clf.predict([[n_height, n_weight, n_shoesize]])

print('new person is ', prediction, ' ?')
