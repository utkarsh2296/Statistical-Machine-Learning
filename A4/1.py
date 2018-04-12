import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from A4.neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt

def split_data(x, y):
    train_x = []
    test_x = []
    train_y = []
    test_y = []

    for i in range(len(y)):
        if i % 2 == 0:
            test_x.append(x[i])
            test_y.append(y[i])
        else:
            train_x.append(x[i])
            train_y.append(y[i])

    return np.array(train_x), np.array(test_x), \
           np.array(train_y), np.array(test_y)


def missing_handle(data):
    missing_removed_data = []

    for index in range(data.shape[0]):
        row = data[index, :]
        row_combined = ','.join(row)
        if '?' in row_combined:
            # print("deleted")
            pass
        else:
            missing_removed_data.append(row)

    missing_removed_data = np.array(missing_removed_data)
    return missing_removed_data


def category_handle(data, variables, one_hot=True):
    for col_key in variables.keys():
        col = data[:, variables[col_key][0]]
        if variables[col_key][1] == 0:
            col = col.astype('float')
            # print(col)
            col = np.divide(col, np.max(col))

        else:
            name, col = np.unique(col, return_inverse=True)
            if col_key == "label":
                print(name)
            col = np.divide(col, np.max(col))
        # np.insert(file_data, variables[col_key][0], col, axis=1)
        # print("va", variables.get(col_key))
        ind = variables.get(col_key)
        ind = ind[0]
        # print("var now : ", ind)
        data[:, ind] = col

    missing_removed_data = data.astype('float')
    x = missing_removed_data[:, :-1]
    if one_hot:
        y = missing_removed_data[:, -1]
        enc = OneHotEncoder(sparse=False)
        enc.fit(np.array(y).reshape(-1, 1))
        y = enc.transform(np.array(y).reshape(-1, 1))
    else:
        y = missing_removed_data[:, -1]

    return x, y


def vary_predictions(predictions, threshold=0.5):
    new_pred = []
    for i in range(len(predictions)):
        if predictions[i] >= threshold:
            new_pred.append(1)
        else:
            new_pred.append(0)

    return new_pred


def plot_points(prob, actual, thresholds):
    points = []

    for i in range(len(thresholds)):
        temp_actual = actual[:]
        temp_predictions = vary_predictions(prob, thresholds[i])
        conf_matrix = confusion_matrix(actual[:], temp_predictions, labels=[0, 1])
        tpr, fpr, frr = get_TPR_FPR_FRR(conf_matrix)

        points.append([tpr, fpr, frr])

    return np.array(points)


def get_TPR_FPR_FRR(confusion_matrix):
    tpr = float(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    fpr = float(confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1]))
    frr = float(confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[0][0]))

    return tpr, fpr, frr


file_data = np.loadtxt("A4/adult_data.txt", delimiter=',', dtype=str)
# file_data = file_data[:50]

data = missing_handle(file_data)

variables = {'age': [0, 0], 'workclass': [1, 1], 'fnlwgt': [2, 0], 'education': [3, 1], 'education-num': [4, 0],
             'marital-status': [5, 1], 'occupation': [6, 1], 'relationship': [7, 1], 'race': [8, 1], 'sex': [9, 1],
             'capital-gain': [10, 0], 'capital-loss': [11, 0], 'hours-per-week': [12, 0], 'native-country': [13, 1],
             'label': [14, 1]}

x, y = category_handle(data, variables, one_hot=False)
train_x, test_x, train_y, test_y = split_data(x, y)


if __name__ == '__main__':

    nn = NeuralNetwork([train_x.shape[1], 3, 1])
    nn.fit(train_x, train_y, epoch=3)
    predicted = nn.predict(test_x)
    acc = nn.accuracy(predicted, test_y)
    print(acc)


    one_count = 0
    zero_count = 0
    one = 0
    zero = 0
    for i in range(len(test_y)):

        if test_y[i] == 1:
            one += 1
            if test_y[i] == predicted[i]:
                one_count += 1

        if test_y[i] == 0:
            zero += 1
            if test_y[i] == predicted[i]:
                zero_count += 1

    print(float(zero_count/zero))
    print(float(one_count/one))

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    pt = plot_points(nn.predict_prob, test_y, thresholds=thresholds)

    plt.clf()
    plt.title("equal error rate")
    plt.plot(pt[:, 2], label="frr")
    plt.plot(pt[:, 1], label="fpr")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(pt[:, 1], pt[:, 0])
    plt.title("ROC curve for Question 1")
    plt.xlabel("False positive Rate")
    plt.ylabel("True positive Rate")
    plt.show()