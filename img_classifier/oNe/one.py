#import librarys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#add shortcut to tf.contrib.learn
learn = tf.contrib.learn
#set logging to 'ERROR'
tf.logging.set_verbosity(tf.logging.ERROR)

#get dataset
mnist = learn.datasets.load_dataset('mnist')

#out of dataset take training data and training labels
data = mnist.train.images
#format labels as numpy asarray
labels = np.asarray(mnist.train.labels, dtype=np.int32)

#out of dataset take testing data and testing labels
test_data = mnist.test.images
#format labels as numpy asarray
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#display image based on its index
def show(img_index):
    img = test_data[img_index]
    plt.title('Example: %d, Label: %d' % (img_index, test_labels[img_index]))
    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
    plt.show()

#take feature columns
feat_clmns = learn.infer_real_valued_columns_from_input(data)
#create classifier
clf = learn.LinearClassifier(feature_columns=feat_clmns, n_classes=10)
#train classifier with out data and labels
clf.fit(data, labels, batch_size=100, steps=1000)

clf.evaluate(test_data, test_labels)
print(clf.evaluate(test_data, test_labels)['accuracy'])

#predict() has changed. This does not work
prediction = clf.predict(test_data[0])
print(prediction)
