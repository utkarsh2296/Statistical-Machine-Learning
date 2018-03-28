from random import shuffle
import numpy as np
import os
import cv2
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import pickle
from LDA import LDA
from pca import PCA
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def read1(path):
    print("calling for ", path)
    data = []
    label = []
    shape = []

    for file in os.listdir(path):
        if not os.path.isfile(path + '/' + file):
            ret_data, ret_label, ret_shape = read1(path + '/' + file)
            data.extend(ret_data)
            label.extend(ret_label)
            shape.extend(ret_shape)
        else:
            if "Ambient" not in file:
                img = cv2.imread(path + '/' + file, -1)
                img = cv2.resize(img, (50, 50))
                shape.append(np.array(img).shape)
                img = img.flatten()

                data.append(img)
                label.append(int(os.path.basename(path)) - 1)
            else:
                continue

    return np.array(data), np.array(label), np.array(shape)


def read2(path):
    # data = []
    # label = []
    # for file in os.listdir(path):
    #     print(file)
    #     dict = unpickle(path + '/' + file)
    #     print(dict.keys())
    #     keys = sorted(list(dict.keys()))
    #     print(keys)
    #     data.extend(dict.get(b'data'))
    #     label.extend(dict.get(b'labels'))
    # data = np.array(data)
    # label = np.array(label)
    data, label = joblib.load('data.pkl'), joblib.load('label.pkl')
    return data, label


def cross_validation(X, Y, folds=5, split_value=0.3, name="lda"):
    # Y = Y.reshape((len(Y), 1))
    # X = np.hstack((X, Y))
    # part = -1
    #
    # if split:
    #     part = split_value
    # else:
    #     part = int(np.math.ceil(len(X) / folds))
    # scores = []
    #
    # for i in range(folds):
    #     test = np.array(X[i * part: (i + 1) * part])
    #     test = [list(d) for d in test]
    #     train = [np.array(j) for j in X if list(j) not in test]
    #     test = np.array(test)
    #     train = np.array(train)
    #
    #     train_x, train_y = train[:, :-1], train[:, -1]
    #     test_x, test_y = test[:, :-1], test[:, -1]
    #
    #     print(train_x.shape)
    #     print(test_x.shape)

    scores = []
    for fold in range(folds):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, shuffle=True, test_size=split_value)

        if name=="lda":
            lda = LDA()
            lda.fit(train_x, train_y)


            lda_train_x = lda.transform(train_x)
            lda_test_x = lda.transform(test_x)
        else:
            pca = PCA()
            pca.fit(train_x)

            pca_train_x = pca.transform(train_x)
            pca_test_x = pca.transform(test_x)

        '''classifier'''
        lr = LogisticRegression(solver='saga', n_jobs=4)
        lr.fit(train_x, train_y)
        score = lr.score(test_x, test_y)
        scores.append(score)
        print("accuracy on  fold ", fold, " : ", score)

    mean = np.mean(scores)
    std = np.std(scores)
    print("mean accuracy : ", mean)
    print("standard deviation : ", std)

    return mean, std, scores


# def roc(test_y, prob, title="", ):
#     predictions = []
#     scores = []
#
#     for i in range(len(test_y)):
#         if test_y[i] == (np.argmax(prob[i])):
#             predictions.append(1)
#             scores.append(prob[i][np.argmax(prob[i])])
#         else:
#             predictions.append(0)
#             scores.append(prob[i][np.argmax(prob[i])])
#
#     fpr, tpr, tick= roc_curve(predictions, scores, pos_label=1)
#     l = len(tick)
#
#     plt.plot(fpr, tpr)
#     plt.title(title)
#     plt.xlabel("FPR values")
#     plt.ylabel("TPR values")
#     # plt.show()
#     plt.savefig("plots/" + title + ".png")
#     # plt.show()
#
#     return


# if __name__ == '__main__':
#     while(True):
c = int(input("enter your choice : "))
if c == 1:
    d = 1
    X, Y, _ = read1('Face_data')
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, shuffle=True)
else:
    d = 2
    X, Y = read2('cifar-10')
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=1/6, shuffle=True)
    print("data read complete")

train_x = np.asarray(train_x)
test_x = np.asarray(test_x)
train_y = np.asarray(train_y)
test_y = np.asarray(test_y)
print(test_y.shape)
print(np.unique(Y))

type = int(input("pca or lda or both ?"))

if type == 1:
    if d == 2:
        split = float(1/6)
    else:
        split = 0.3

    s = "pca"
    pca = PCA()
    pca.fit(train_x)
    joblib.dump(pca.eigen_vectors, "pca_prjection_" + str(d) + ".pkl")

    pca_train_x = pca.transform(train_x)
    pca_test_x = pca.transform(test_x)
    # del pca
    print("pca done")

    lr = LogisticRegression(solver='saga', n_jobs=4)
    lr.fit(pca_train_x, train_y)
    print("accuracy on test data : ", lr.score(pca_test_x, test_y))
    pr = lr.predict_proba(pca_test_x)
    pt = lr.predict(pca_test_x)

    tt = [np.argmax(i) for i in pr[:10]]
    print(tt)
    print(pt[:10])

    # roc(test_y, pr, title=s + " on dataset " + str(d))


    # '''cross validation'''
    # proj_X = pca.transform(X)
    # mean, std, scores = cross_validation(proj_X, Y)
    # print(mean, std, scores)

elif type == 2:
    if d == 2:
        split = float(1/6)
    else:
        split = 0.3
    s = "lda"
    lda = LDA()
    lda.fit(train_x, train_y)
    # print(lda.eigen_vectors.shape)
    joblib.dump(lda.eigen_vectors, "lda_projection_" + str(d) + ".pkl")

    '''classifier'''
    lda_train_x = lda.transform(train_x)
    lda_test_x = lda.transform(test_x)

    lr = LogisticRegression(solver='saga', n_jobs=4)
    lr.fit(lda_train_x, train_y)
    print("lda accuracy : ", lr.score(lda_test_x, test_y))

    '''roc'''
    pr = lr.predict_proba(lda_test_x)
    roc(test_y, pr, title=s + " on dataset " + str(d))

    # '''cross validation'''
    # proj_X = lda.transform(X)
    # mean, std, scores = cross_validation(proj_X, Y)
    # print(mean, std, scores)


elif type == 3:
    s = "pca_lda"
    pca = PCA()
    pca.fit(train_x)

    pca_train_x = pca.transform(train_x)
    pca_test_x = pca.transform(test_x)

    lda = LDA()
    lda.fit(pca_train_x, train_y)
    # print(lda.eigen_vectors.shape)

    '''classifier'''
    lda_train_x = lda.transform(pca_train_x)
    lda_test_x = lda.transform(pca_test_x)

    lr = LogisticRegression(solver='saga', n_jobs=4, verbose=True)
    lr.fit(lda_train_x, train_y)
    print("pca --> lda : ", lr.score(lda_test_x, test_y))
    # print(test_x.shape)

    '''roc'''
    pr = lr.predict_proba(lda_test_x)
    roc(test_y, pr, title=s + " on dataset " + str(d))

else:
    s = "lda_pca"

    lda = LDA()
    lda.fit(train_x, train_y)
    # print(lda.eigen_vectors.shape)

    lda_train_x = lda.transform(train_x)
    lda_test_x = lda.transform(test_x)

    pca = PCA()
    pca.fit(lda_train_x)

    pca_train_x = pca.transform(lda_train_x)
    pca_test_x = pca.transform(lda_test_x)

    lr = LogisticRegression(solver='saga', n_jobs=4, verbose=True)
    lr.fit(pca_train_x, train_y)
    print("lda --> pca : ", lr.score(pca_test_x, test_y))
    # print(test_x.shape)

    '''roc'''
    pr = lr.predict_proba(pca_test_x)
    roc(test_y, pr, title=s + " on dataset " + str(d))
