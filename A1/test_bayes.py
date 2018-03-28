import numpy as np

from train_bayes import train


def predict(data, likelihoods, priors=[0.5, 0.5]):
    # likelihood_1 == conditional_prob1
    # likelihood_2 == conditional_prob2
    likelihood_1 = likelihoods[1]
    likelihood_2 = likelihoods[2]

    posteriors = []
    roc_values = []

    for i in data:
        outcome_likelihood_1 = []
        outcome_likelihood_2 = []

        for j in range(len(i) - 1):
            value = i[j]
            temp_likelihood, count = likelihood_1[j][0], likelihood_1[j][1]

            if j == 4:
                classes = [0, 10, 20, 30, 40, 50, 60, 70]
                value = np.digitize(i[j], classes)

            if value in list(temp_likelihood.keys()):
                outcome_likelihood_1.append(temp_likelihood[int(value)])
            else:
                prob = float(1 / (count + len(temp_likelihood.keys()) + 1))
                outcome_likelihood_1.append(prob)

        for j in range(len(i) - 1):
            value = i[j]
            temp_likelihood, count = likelihood_2[j][0], likelihood_2[j][1]

            if j == 4:
                classes = [0, 10, 20, 30, 40, 50, 60, 70]
                value = np.digitize(i[j], classes)

            if value in list(temp_likelihood.keys()):
                outcome_likelihood_2.append(temp_likelihood[int(value)])
            else:
                prob = float(1 / (count + len(temp_likelihood.keys())))
                outcome_likelihood_2.append(prob)

        posterior = np.argmax([np.prod(np.array(outcome_likelihood_1)), np.prod(np.array(outcome_likelihood_2))])
        if posterior == 0:
            posteriors.append(1)
        else:
            posteriors.append(3)

        # for ROC curve
        roc_prob = [np.prod(np.array(outcome_likelihood_1)), np.prod(np.array(outcome_likelihood_2))]
        roc_prob = [float(i / (np.sum(roc_prob))) for i in roc_prob]

        roc_values.append(roc_prob)

    return posteriors, roc_values


def accuracy(real, predicted):
    count = 0
    for i in range(len(real)):
        if real[i] == predicted[i]:
            count += 1

    return float(count / len(real))


def cross_validation(data, folds=5):
    part = int(np.math.ceil(len(data) / folds))
    accuracies = []

    for i in range(folds):
        test_data = np.array(data[i * part: (i + 1) * part])
        test_data = [list(d) for d in test_data]
        train_data = [np.array(j) for j in data if list(j) not in test_data]
        test_data = np.array(test_data)
        train_data = np.array(train_data)

        _, likelihoods = train(train_data)
        output, _ = predict(test_data, likelihoods=likelihoods)

        accuracies.append(accuracy(output, test_data[:, -1]))

    mean = np.mean(accuracies)
    std = np.std(accuracies)

    return mean, std


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
