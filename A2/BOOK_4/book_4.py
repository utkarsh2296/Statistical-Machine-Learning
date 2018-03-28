import numpy as np

c0 = np.array([[-5.01, -8.12, -3.68], [-5.43, -3.48, -3.54],
               [1.08, -5.52, 1.66], [0.86, - 3.78, -4.11],
               [-2.67, 0.63, 7.39], [4.94, 3.29, 2.08],
               [-2.51, 2.09, -2.59], [-2.25, -2.13, -6.94],
               [5.56, 2.86, -2.26], [1.03, -3.33, 4.33]])

c1 = np.array([[-0.91, -0.18, -0.05], [1.30, -2.06, -3.53], [-7.75, -4.54, -0.95], [-5.47, 0.50, 3.92], [6.14, 5.72, -4.85],
               [3.60, 1.26, 4.36], [5.37, -4.63, -3.65], [7.18, 1.46, -6.66], [-7.39, 1.17, 6.30], [-7.50, -6.32, -0.31]])

c2 = np.array([[5.35, 2.26, 8.13], [5.12, 3.22, -2.66], [-1.34, -5.31, -9.87], [4.48, 3.42, 5.19], [7.11, 2.39, 9.21],
               [7.17, 4.33, -0.98], [5.75, 3.97, 6.65], [0.77, 0.27, 2.41], [0.90, -0.43, -8.71], [3.52, -0.36, 6.43]])

means = [0] * 3
covs = [0] * 3
priors = [1 / 3] * 3
new_priors = [0.8, 0.1, 0.1]

means[0] = [np.mean(c0[:, i]) for i in range(c0.shape[1])]
means[1] = [np.mean(c1[:, i]) for i in range(c1.shape[1])]
means[2] = [np.mean(c2[:, i]) for i in range(c2.shape[1])]

covs[0] = np.cov(c0.T)
covs[1] = np.cov(c1.T)
covs[2] = np.cov(c2.T)

'''points to classify'''
points = np.array([[1, 2, 1], [5, 3, 2], [0, 0, 0], [1, 0, 0]])


def mahanobolis_based_decision(points, means, covs, priors):
    g_x = []
    mahan = []

    for i in range(len(means)):
        # print(points)
        # print("mean : ", means[i])
        x_sub = np.subtract(points, means[i])
        inv = np.linalg.inv(covs[i])
        det = np.linalg.det(covs[i])
        mahanoboli = (np.diagonal(np.dot(np.dot(x_sub, inv), x_sub.T)))
        mahan.append(np.sqrt(mahanoboli))
        # print("maha : ", mahanoboli)
        # print("extras : ", np.log(np.linalg.det(covs[i])))
        g_i = (-0.5) * mahanoboli + np.log(priors[i]) - (0.5) * np.log(np.linalg.det(covs[i]))
        g_x.append(g_i)
        # print(g_i)

    g_x = np.array(g_x)
    print(g_x)
    # print(np.array(mahan), "adbhgasbhchvh")
    mahan_index = np.argmin(mahan, axis=0)
    # print("index : ", mahan_index)
    max_index = np.argmax(g_x, axis=0)
    return max_index, mahan, mahan_index


def printX(points, prior):
    print("--------------------------------------------------")
    print("when prior = ", prior)
    classes, mahan , mahan_index= mahanobolis_based_decision(points, means, covs, prior)
    for i in range(len(points)):
        print("--------------------------------------------------")
        classes, mahan, mahan_index= mahanobolis_based_decision(points, means, covs, prior)
        print("for point ", points[i], " : ")
        print("mahanolobis distance from class 1 : ", mahan[0][i])
        print("mahanolobis distance from class 2 : ", mahan[1][i])
        print("mahanolobis distance from class 3 : ", mahan[2][i])
        print("assigned class to the point : ", classes[i] + 1)
        print("assigned class to the point on the basis of mahanobolis distance : ", mahan_index[i] + 1)
        print("--------------------------------------------------")


printX(points, priors)
printX(points, new_priors)

