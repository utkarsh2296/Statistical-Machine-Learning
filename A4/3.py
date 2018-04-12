import os
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def read_data(path="F:\IIITD\Semester_2\Statistical Machine Learning\A4\dataset\cifar-10"):
    data = []
    label = []
    for file in os.listdir(path):
        if os.path.isfile(path + '/' + file):
            d = unpickle(path + '/' + file)
            print(d.keys())
            data.extend(d.get(b'data'))
            label.extend(d.get(b'labels'))



    data = [d[:1024] for d in data]
    return np.array(data), np.array(label)

d, l = read_data()

train_x, test_x, train_y, test_y = train_test_split(d, l, test_size=0.3, shuffle=True)

nn = MLPClassifier((50))
bag = BaggingClassifier(base_estimator=nn, n_estimators=15, verbose=True)
bag.fit(train_x, train_y)
bag.score(test_x, test_y)

'''NN'''
nn = MLPClassifier((400, 150), early_stopping=True)
nn.fit(train_x, train_y)
print(nn.score(test_x, test_y))


import matplotlib.pyplot as plt

plt.title("bagging on CIFAR-10")
plt.plot([0.273, 0.289, 0.328], label="n_estimators")
plt.plot([0.255, 0.255, 0.255], label="neural network only")
plt.ylabel("test accuracy")
plt.legend()
plt.xticks(np.arange(3), [5, 10, 15])
plt.show()