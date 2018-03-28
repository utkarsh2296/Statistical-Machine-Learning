
import numpy as np
from sklearn.externals import joblib

c0 = np.array(
    [[-5.01, -8.12, -3.68],
     [-5.43, -3.48, -3.54],
     [1.08, -5.52, 1.66],
     [0.86, - 3.78, -4.11],
     [-2.67, 0.63, 7.39],
     [4.94, 3.29, 2.08],
     [-2.51, 2.09, -2.59],
     [-2.25, -2.13, -6.94],
     [5.56, 2.86, -2.26],
     [1.03, -3.33, 4.33]])

c1 = np.array(
    [[-0.91, -0.18, -0.05],
     [1.30, -2.06, -3.53],
     [-7.75, -4.54, -0.95],
     [-5.47, 0.50, 3.92],
     [6.14, 5.72, -4.85],
     [3.60, 1.26, 4.36],
     [5.37, -4.63, -3.65],
     [7.18, 1.46, -6.66],
     [-7.39, 1.17, 6.30],
     [-7.50, -6.32, -0.31]])

c2 = np.array(
    [[5.35, 2.26, 8.13],
     [5.12, 3.22, -2.66],
     [-1.34, -5.31, -9.87],
     [4.48, 3.42, 5.19],
     [7.11, 2.39, 9.21],
     [7.17, 4.33, -0.98],
     [5.75, 3.97, 6.65],
     [0.77, 0.27, 2.41],
     [0.90, -0.43, -8.71],
     [3.52, -0.36, 6.43]])


priors = [0.5, 0.5, 0]
means = [0] * 3
covs = [0] * 3

means[0] = [np.mean(c0[:, i]) for i in range(c0.shape[1])]
means[1] = [np.mean(c1[:, i]) for i in range(c1.shape[1])]
means[2] = [np.mean(c2[:, i]) for i in range(c2.shape[1])]

covs[0] = np.cov(c0.T)
covs[1] = np.cov(c1.T)
covs[2] = np.cov(c2.T)

X = []
Y = []
classes = [c0, c1]
for i in range(len(classes)):
    for j in classes[i]:
        X.append(j)
        Y.append(i)


def discriminant_fnction(data, mean, covariance, prior, features=[0]):
    data = np.array([data[i] for i in features])
    data = np.reshape(data, (1, len(data)))
    # print("data : ", data)
    mean = np.array([mean[i] for i in features])
    # print("mean : ", mean)
    idx = np.ix_(features, features)
    cov = covariance[idx]
    # print("cov : ", cov)
    sub_x = np.subtract(data, mean)
    # print("sub_x : ", sub_x)
    # print("done4")
    inv_matrix = np.linalg.inv(cov)
    # print("inv matrix : ", inv_matrix)
    # print("done5")
    # print(data.shape)
    # print(inv_matrix.shape)
    # print("ist term : ", (-0.5) * np.dot(np.dot(sub_x, inv_matrix), sub_x.T))
    # print("2nd term : ", 0.5 * np.log(np.linalg.det(cov)))
    # print("3rd term : ", np.log(prior))
    g_x = (-0.5) * np.dot(np.dot(sub_x, inv_matrix), sub_x.T) - 0.5 * np.log(np.linalg.det(cov)) + np.log(prior)
    # print("gx = ", g_x[0][0])
    return g_x[0][0]


def bhattacharya_bound(mean1, mean2, covariance1, covariance2, prior1, prior2, features=[0]):
    mean1 = np.array([mean1[i] for i in features])
    np.reshape(mean1, (1, len(mean1)))
    idx = np.ix_(features, features)
    cov1 = covariance1[idx]

    # print("cov1 : ", cov1)

    mean2 = np.array([mean2[i] for i in features])
    np.reshape(mean2, (1, len(mean2)))
    idx = np.ix_(features, features)
    cov2 = covariance2[idx]

    # print("cov2 : ", cov2)
    # print("cov average : ", (cov1 + cov2)/2)
    # print(mean1.shape)
    # print(cov1.shape)
    # print("mean differe : ", np.subtract(mean2, mean1))
    # print("inv : ", np.linalg.inv((cov1 + cov2) / 2))
    # print("avg det : ", (np.linalg.det((cov1 + cov2) / 2)) / (np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))))

    k = float(1 / 8) * np.dot(np.dot(np.subtract(mean2, mean1), np.linalg.inv((cov1 + cov2) / 2)),
                              np.subtract(mean2, mean1).T) + (0.5) * np.log(
        (np.linalg.det((cov1 + cov2) / 2)) / (np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))))
    bound = np.sqrt(prior1 * prior2) * np.power(np.e, -k)

    return bound


feature = []
for f in range(3):
    feature.append(f)
    out = []

    for i in X:
        g1_x = discriminant_fnction(i, means[0], covs[0], priors[0], features=feature)
        g2_x = discriminant_fnction(i, means[1], covs[1], priors[1], features=feature)

        if g1_x - g2_x > 0:
            out.append(0)
        else:
            out.append(1)

    c = 0

    for i in range(len(out)):
        if out[i] != Y[i]:
            c += 1

    print("emperical error with " + str(len(feature)) + " features : ", float(c / len(out)))
    b = bhattacharya_bound(means[0], means[1], covs[0], covs[1], priors[0], priors[1], features=feature)
    print("bhattacharya bound with " + str(len(feature)) + " features : ", b)


