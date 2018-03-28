# 1. Whether of not the TA is a native English speaker (binary); 1=English speaker, 2=non-English speaker
# 2. Course instructor (categorical, 25 categories)
# 3. Course (categorical, 26 categories)
# 4. Summer or regular semester (binary) 1=Summer, 2=Regular
# 5. Class size (numerical)
# 6. Class attribute (categorical) 1=Low, 2=Medium, 3=High
import joblib
import os
import numpy as np
from numpy.random import shuffle
from test_bayes import predict, accuracy, cross_validation, vary_predictions, plot_points, confusion_matrix, getTPR_FPR
from train_bayes import train
import matplotlib.pyplot as plt
# import scipy
# import sklearn


def read(file):
    data_from_file = np.genfromtxt(file, delimiter=',')
    data = []
    for i in data_from_file:
        if i[-1] == 2:
            continue
        else:
            data.append(i)

    return np.array(data)


def split_train_test(data, percent=0.7):
    shuffle(data)
    ratio = int(percent * len(data))

    train_data = data[:ratio, :]
    test_data = data[ratio:, :]

    return train_data, test_data


def plot(x, y, title = "", xlabel="", ylabel="", type="plot"):
    if type == "plot":
        plt.plot(x, y)
    elif type == "scatter":
        plt.scatter(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


'''READ DATA'''
data = read('tae.data')

'''SPLIT DATA'''
# train_t, test_t = split_train_test(data, 0.7)
# joblib.dump(train_t, 'train.data')
# joblib.dump(test_t, 'test.data')

likelihood_probabilities = []
x = joblib.load('C:\Users\Vaibhav\PycharmProjects\SML\Assignment1\model.sav')
'''TRAINING AND SAVING MODEL'''
if os.path.isfile('model.sav'):
    print("nonono")
    likelihood_probabilities = joblib.load('model.sav')
    test_t = joblib.load('test.data')
else:
    train_t, test_t = split_train_test(data, 0.7)
    prior_probabilities, likelihood_probabilities = train(train_t)
    print("hahaha")
    joblib.dump(likelihood_probabilities, 'model.sav')
    joblib.dump(train_t, 'train.data')
    joblib.dump(test_t, 'test.data')

# print(likelihood_probabilities)
'''TESTING'''
predicted, roc_values = predict(test_t, likelihood_probabilities)
# predicted = vary_predictions(predicted)

'''ACCURACY'''
actual = test_t[:, -1]
print("accuracy : ", accuracy(actual, predicted))

'''CONFUSION MATRIX'''
cm = confusion_matrix(actual, predicted)
print("confusion matrix : ")
print(cm)

norm_cm = np.array([np.array(i / np.sum(i)) for i in cm])
print(norm_cm)
plt.matshow(norm_cm)
plt.title("confusion matrix")
plt.xlabel("high class")
plt.ylabel("low class")
plt.colorbar()
plt.show()

tpr, far = getTPR_FPR(cm)
print("TPR : ", tpr, "||  FAR", far)

mean, std = cross_validation(data)
print("mean : ", mean)
print("std : ", std)

thresholds = np.arange(0, 1, step=0.1)
print(thresholds)
points = plot_points(roc_values, actual[:], thresholds)

print(points)
points = np.array(points)

plot(points[:, 1], points[:, 0], title='ROC curve')

import numpy as np




data = read('')

print(1)
