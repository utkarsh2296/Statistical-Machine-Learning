import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def vary_predictions(predictions, threshold=0.5):
    new_pred = []
    for i in range(len(predictions)):
        if predictions[i][1] >= threshold:
            new_pred.append(3)
        else:
            new_pred.append(1)

    return new_pred


def confusion_matrix(actual, predicted, classes=2):
    # print("actual before : ", actual)
    new_actual = []

    for i in range(len(actual)):
        if actual[i] == 1:
            # actual[i] = 0
            new_actual.append(0)
        elif actual[i] == 3:
            # actual[i] = 1
            new_actual.append(1)

        if predicted[i] == 1:
            predicted[i] = 0
        elif predicted[i] == 3:
            predicted[i] = 1

    conf_matrix = np.zeros((classes, classes))
    # print("actual after : ", actual)
    # print("predicted : ", predicted)

    for loop in range(len(predicted)):
        # i, j = int(actual[loop]), int(predicted[loop])
        i, j = int(new_actual[loop]), int(predicted[loop])
        conf_matrix[i][j] += 1

    return conf_matrix


def plot_points(roc_values, actual, thresholds):
    points = []
    # print("length : ", len(roc_values))

    for i in range(len(thresholds)):
        temp_actual = actual[:]
        # print("temp actual : ", temp_actual)
        # print("threshold : ", thresholds[i])
        temp_predictions = vary_predictions(roc_values, thresholds[i])
        # print("temp predictions : ", temp_predictions)
        conf_matrix = confusion_matrix(actual[:], temp_predictions)
        # print("confusion matrix : ", conf_matrix)
        tpr, fpr = getTPR_FPR(conf_matrix)

        points.append([tpr, fpr])

    return points


def getTPR_FPR(confusion_matrix):
    tpr = float(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
    fpr = float(confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1]))

    return tpr, fpr


# def roc(test_y, prob, title="", ):
#     predictions = []
#     scores = []
#
#     for i in range(len(test_y)):
#         if test_y[i] == (np.argmax(prob[i])):
#             predictions.append(0)
#             scores.append(prob[i][np.argmax(prob[i])])
#         else:
#             predictions.append(1)
#             scores.append(prob[i][np.argmax(prob[i])])
#
#     thresholds = np.arange(0, 1, 0.1)
#     plot_points(scores, predictions, thresholds)
#
#     # fpr, tpr, tick= roc_curve(predictions, scores, pos_label=1)
#     # l = len(tick)
#
#     # plt.plot(fpr, tpr)
#     # plt.title(title)
#     # plt.xlabel("FPR values")
#     # plt.ylabel("TPR values")
#     # # plt.show()
#     # plt.savefig("plots/" + title + ".png")
#     # # plt.show()
#
#     return