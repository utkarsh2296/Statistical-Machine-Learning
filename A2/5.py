import idx2numpy
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from train_model import NaiveBayes

train_x = idx2numpy.convert_from_file('C:\\Users\Vaibhav\PycharmProjects\SML\A2\\train-images.idx3-ubyte')
train_y = idx2numpy.convert_from_file('C:\\Users\Vaibhav\PycharmProjects\SML\A2\\train-labels.idx1-ubyte')

test_x = idx2numpy.convert_from_file('C:\\Users\Vaibhav\PycharmProjects\SML\A2\\t10k-images.idx3-ubyte')
test_y = idx2numpy.convert_from_file('C:\\Users\Vaibhav\PycharmProjects\SML\A2\\t10k-labels.idx1-ubyte')

train_x = np.array([d.flatten() for d in train_x])
test_x = np.array([d.flatten() for d in test_x])

nb = NaiveBayes()
nb.fit(train_x, train_y, one_hot=False, classes=[0, 1], check=False)
predicted, actual, score = nb.predict(test_x, test_y, classes=[0, 1], check=False, prior=[0.5, 0.5])
count = 0

for i in range(len(actual)):
    if actual[i] == predicted[i]:
        count += 1

print("accuracy is : ", (100 * count) / len(actual))
# # # ROC CURVE
# _, actual = np.unique(actual, return_inverse=True)
# _, predicted = np.unique(predicted, return_inverse=True)
# print(len(actual), len(predicted))
# fpr, tpr, thresholds = roc_curve(actual, score, pos_label=0)
#
# plt.plot(fpr, tpr)
# plt.title("ROC curve for 0 and 1 normal ")
# plt.xlabel("fpr values")
# plt.ylabel("tpr values")
# plt.savefig('Q5/' + 'roc_01_org.png')
# plt.show()
