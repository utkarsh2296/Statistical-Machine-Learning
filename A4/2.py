import numpy as np
import idx2numpy

from sklearn.neural_network import MLPClassifier

x_train = idx2numpy.convert_from_file("emnist-balanced-train-images-idx3-ubyte")
y_train = idx2numpy.convert_from_file("emnist-balanced-train-labels-idx1-ubyte")
x_test = idx2numpy.convert_from_file("emnist-balanced-test-images-idx3-ubyte")
y_test = idx2numpy.convert_from_file("emnist-balanced-test-labels-idx1-ubyte")

x_train = [i.flatten() for i in x_train]
x_test = [i.flatten() for i in x_test]

x_train = np.array(x_train)
x_test = np.array(x_test)

activation = str(input("enter activation function: "))

print("-" * 30)
print("exteriment with activation function : ", activation)
print("-" * 30)

'''part (a)'''
print("-" * 30)
print("part (a) ")
print("-" * 30)
lr = [0.001, 0.1, 0.2]
acc = []
for l in lr:
    print("lr = ", l)
    nn = MLPClassifier((256, 128, 64), activation=activation, learning_rate_init=l, early_stopping=True)
    nn.fit(x_train, y_train)
    acc.append(nn.score(x_test, y_test))
    print("accuracy with learning rate", l)

print(acc)  #[0.8077659574468085, 0.5881382978723404, 0.02127659574468085, 0.02127659574468085]

'''part(b)'''
print("-" * 30)
print("part (b) ")
print("-" * 30)
nn_1 = MLPClassifier((350, 700, 350), learning_rate_init=0.1, max_iter=100, activation=activation, early_stopping=True)
nn_1.fit(x_train, y_train)
print("accuracy with follwoing hidden layers : (350, 700, 350) : ", nn_1.score(x_test, y_test))

nn_2 = MLPClassifier((50, 100, 200), learning_rate_init=0.1, max_iter=100, activation=activation, early_stopping=True)
nn_2.fit(x_train, y_train)
print("accuracy with follwoing hidden layers : (64, 256, 128, 64, 128) : ", nn_2.score(x_test, y_test))

'''part (c)'''
print("-" * 30)
print("part (c) ")
print("-" * 30)
nn_1 = MLPClassifier((256, 128, 64, 128), learning_rate_init=0.1, max_iter=100, activation=activation, early_stopping=True)
nn_1.fit(x_train, y_train)
print("accuracy with adding H4 hidden layers : (256, 128, 64, 128) : ", nn_1.score(x_test, y_test))

'''part (d)'''
print("-" * 30)
print("part (d) ")
print("-" * 30)
mi = [25, 30, 40, 50, 60, 70, 80, 90]
mi_acc = []
for i in mi:
    nn_4 = None
    nn_4 = MLPClassifier((256, 128, 64), learning_rate_init=0.1, max_iter=i, batch_size=40, activation=activation, early_stopping=True)
    nn_4.fit(x_train, y_train)
    mi_acc.append(nn_4.score(x_test, y_test))
    print("accuracy at max iteration " + str(i) + " : ", nn_4.score(x_test, y_test))

print("accuracy : ", mi_acc)


'''part 6'''

nn = MLPClassifier(hidden_layer_sizes=(600, 300, 150, 75, 37), solver='sgd', learning_rate='invscaling', early_stopping=True)
nn.fit(x_train, y_train)
print(nn.score(x_test, y_test))

alphas = [0.01, 0.008, 0.005 ]
for a in alphas:

    nn = MLPClassifier(hidden_layer_sizes=(200, 400, 200), solver='sgd', alpha=a, early_stopping=True)
    nn.fit(x_train, y_train)
    print("alpha : ", a)
    print(nn.score(x_test, y_test))



# import matplotlib.pyplot as plt
#
#
# # acc_relu =
#
# tick = [25, 30, 40, 50, 60, 70, 80, 90]
#
#
#
# plt.title("accuracy at different learning rate")
# plt.plot(acc_identity, label="identity activation")
# # plt.plot(acc_relu, label="relu activation")
# # plt.plot(acc_sig, label="sigmoid activation")
# plt.xticks(np.arange(len(tick)), tick)
# plt.legend()
# plt.show()
#
