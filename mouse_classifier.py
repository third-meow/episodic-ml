# Mouse Classifier takes the price, weight and number of LEDs
# found on a computer mouse and figures out if it is a 'gaming'
# mouse or just a 'normal' mouse

from sklearn import tree

#mice data follows format [price, num of LEDs, weight]
mice = [[76, 3, 100],
        [5, 0, 70],
        [200, 9, 300],
        [40, 2, 90],
        [90, 5, 150],
        [13, 1, 78],
        [70, 4, 180],
        [19, 1, 100]]

labels = ['gaming','normal','gaming','normal','gaming','normal','gaming','normal']

clf = tree.DecisionTreeClassifier()

clf= clf.fit(mice, labels)

new_price = int(input('price of mouse: '))
new_leds = int(input('number of leds found on mouse: '))
new_weight = int(input('weight of mouse(grams): '))

pre = clf.predict([[new_price, new_leds, new_weight]])
 
print(pre)
