import idx2numpy
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.means = {}
        self.sd = {}

    def fit(self, train_data, train_label, one_hot=False, classes=[], check=False):
        if one_hot is True:
            train_label = np.argmax([np.argmax(row) for row in train_label])

        if check:
            print("taking 10% class 3 and 90 % class 8 data for train : ")
            train_data, train_label = self.split(train_data, train_label, classes)

        train_data = np.insert(train_data, len(train_data[0]), train_label, axis=1)
        columns = train_data.shape[1] - 1
        print("columns : ", columns)

        for c in classes:
            print("class is :", c)
            class_data = np.array([row for row in train_data if row[-1] == c])

            mean_vector = np.array([np.mean(class_data[:, i]) for i in range(columns)])
            std_vector = np.array([np.std(class_data[:, i]) for i in range(columns)])
            self.means[c] = mean_vector
            self.sd[c] = std_vector

    def predict(self, test_data, test_label, prior=[0.5, 0.5], one_hot=False, classes=None, check=False):
        if classes is None:
            classes = []
        if one_hot is True:
            test_label = np.argmax([np.argmax(row) for row in test_label])

        if check:
            test_data, test_label = self.split(test_data, test_label, classes)

        else:

            new_t = []
            new_l = []

            for i in range(len(test_data)):
                if test_label[i] in classes:
                    new_t.append(test_data[i])
                    new_l.append(test_label[i])

            # new_t = np.array(new_t)
            # new_l = np.array(new_l)

            test_data = new_t[:]
            test_label = new_l[:]

        print("haha")
        test_data = np.insert(test_data, len(test_data[0]), test_label, axis=1)
        print("haha done")
        columns = test_data.shape[1] - 1
        print("test shape: ", test_data.shape)
        print("columns : ", columns)
        result = []

        score = []
        for data in test_data:
            if data[-1] in classes:
                data = np.array(data[:-1])
                # print("now shape: ", data.shape)
                likelihood = []
                p = 0
                for c in classes:
                    mean = self.means[c]
                    sd = self.sd[c]
                    sd = np.add(sd, 0.0001)
                    const = np.sqrt(2 * np.pi)
                    sd_log = np.log(1 / (np.multiply(const, sd)))
                    # print("sd_log: ", sd_log)

                    factor = np.divide(np.square(np.subtract(data, mean)), np.multiply(2, np.square(sd)))
                    likelihood.append(np.sum(sd_log) - np.sum(factor) + np.log(prior[p]))
                    p+=1
                score.append(likelihood[0] - likelihood[1])
                max_index = np.argmax(likelihood)
                result.append(classes[max_index])

        return result, test_label, score

    def split(self, X, Y, classes = [], class1=0.1, class2=0.9):
        label, count = np.unique(Y, return_counts=True)
        label = [i for i in label]
        count = [i for i in count]
        len1 = count[list(label).index(3)]
        len2 = count[list(label).index(8)]

        len1 = class1 * len1
        len2 = class2 * len2

        l1_count = 0
        l2_count = 0

        data_x = []
        data_y = []

        for i in range(len(X)):
            if Y[i] == classes[0]:
                if l1_count == len1:
                    continue
                else:
                    l1_count += 1
                    data_x.append(X[i])
                    data_y.append(Y[i])

            elif Y[i] == classes[1]:
                if l2_count == len2:
                    continue
                else:
                    l2_count += 1
                    data_x.append(X[i])
                    data_y.append(Y[i])

        cl, cn = np.unique(data_y, return_counts=True)
        print("class : ", cl)
        print("count : ", cn)

        return np.array(data_x), np.array(data_y)
