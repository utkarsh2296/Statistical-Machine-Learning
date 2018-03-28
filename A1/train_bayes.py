import numpy as np


def train(data):
    classes, count = np.unique(data[:, -1], return_counts=True)
    prior_probability = [float(i / np.sum(count)) for i in count]

    data_class1 = np.array([i for i in data if i[-1] == 1])
    data_class2 = np.array([i for i in data if i[-1] == 3])

    conditional_probs1 = []
    conditional_probs2 = []

    for i in range(5):
        keys, count = np.unique(data_class1[:, i], return_counts=True)

        if i == 4:
            data_4 = data_class1[:, -2]
            classes = [0, 10, 20, 30, 40, 50, 60, 70]
            bins = np.digitize(data_4, classes)
            keys, count = np.unique(bins, return_counts=True)

        temp_conditional_probs = [float((j+1) / (np.sum(count) + len(keys) + 1)) for j in count]
        prob_dict = tuple([dict(zip(keys, temp_conditional_probs)), np.sum(count)])
        conditional_probs1.append(prob_dict)

    for i in range(5):
        keys, count = np.unique(data_class2[:, i], return_counts=True)

        if i == 4:
            data_4 = data_class2[:, -2]
            classes = [0, 10, 20, 30, 40, 50, 60, 70]
            bins = np.digitize(data_4, classes)
            keys, count = np.unique(bins, return_counts=True)

        temp_conditional_probs = [float((j+1) / (np.sum(count) + len(keys) + 1)) for j in count]
        prob_dict = tuple([dict(zip(keys, temp_conditional_probs)), np.sum(count)])
        conditional_probs2.append(prob_dict)

    likelihoods = [conditional_probs1, conditional_probs2]
    classes = [1,2]
    likelihoods = dict(zip(classes, likelihoods))

    return prior_probability, likelihoods