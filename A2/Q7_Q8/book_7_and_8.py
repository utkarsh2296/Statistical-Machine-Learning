import numpy as np
from scipy.stats import norm, multivariate_normal
import math
import matplotlib.pyplot as plt


def generate(mu, sigma, data_points=10):
    # print(mu[1])
    data_1 = np.random.multivariate_normal([mu[0]], [[sigma[0]]], data_points)
    data_2 = np.random.multivariate_normal([mu[1]], [[sigma[1]]], data_points)

    return data_1, data_2


def discriminant_fnction(data, mean, covariance, prior, features=[0], same_var=False):
    if same_var == True:
        g_x = np.multiply(float(1 / covariance) * mean, data) + np.log(prior) - float(1 / (2 * covariance)) * np.square(
            mean)
        return g_x
    else:
        # print("done1")
        mean = np.array([mean[i] for i in features])
        sub_x = np.subtract(data, mean)
        # print("sub_x", sub_x)
        # print("done4")
        # print("sub x shape : ", sub_x.shape)
        # print("asdhcvsgfdcvsahvc", np.log(1))
        g_x = (-0.5) * np.square(sub_x) - 0.5 * np.log(covariance) + np.log(prior)
        # print("gx = ", g_x)
    return g_x


def bhattacharya_bound(mean1, mean2, cov1, cov2, prior1, prior2, features=[0]):
    # print("term1 : ", np.subtract(mean1, mean2))
    # print("(cov1 + cov2) / 2) : ", (cov1 + cov2) / 2)
    # print("term 3 : ", np.log(
    #     ((cov1 + cov2) / 2) / (np.sqrt(cov1 * cov2))))
    k = float(1 / 8) * np.square(np.subtract(mean1, mean2)) * 1 / ((cov1 + cov2) / 2) + (0.5) * np.log(
        ((cov1 + cov2) / 2) / (np.sqrt(cov1 * cov2)))
    bound = np.sqrt(prior1 * prior2) * np.power(np.e, -k)

    return bound


def classify(d1, d2, mu, sigma, prior, count):
    Y = [0] * count + [1] * count
    data = np.array(list(d1) + list(d2))
    g1 = discriminant_fnction(data, [mu[0]], sigma[0], prior=0.5)
    g2 = discriminant_fnction(data, [mu[1]], sigma[1], prior=0.5)
    g1 = np.array(g1).flatten()
    g2 = np.array(g2).flatten()

    output = []
    for i in range(len(g1)):
        if g1[i] > g2[i]:
            output.append(0)
        else:
            output.append(1)

    count = 0

    for i in range(len(Y)):
        if Y[i]!=output[i]:
            count +=1

    emp_err = float(count/len(Y))

    return output, Y, emp_err


def square_root(mu, sigma, priors):
    A = 0.5 * float((1 / sigma[0]) - float(1 / sigma[1]))
    B = -(float((mu[0]) / sigma[0])) + float((mu[1]) / sigma[1])
    C = 0.5 * (float(np.square(mu[0]) / sigma[0]) - float(np.square(mu[1]) / sigma[1])) - 0.5 * float(
        np.log(sigma[1] / sigma[0])) + float(np.log(
        priors[1] / priors[0]))
    p = [A, B, C]
    # print("A : ", A)
    # print("B : ", B)
    # print("C : ", C)
    roots = np.roots(p)
    # print("roots : ", roots)
    return roots


def region(x, mu, sigma, prior):
    print("norm : ", norm.pdf(x, mu, sigma))
    ret = float(1/np.sqrt(2*np.pi*sigma))*np.power(np.e, -(np.square(x-mu))/(2*sigma))*prior
    # ret = print("value : ", norm.pdf(x, mu, sigma) * prior, "mean : ", mu, "cov : ", sigma)
    # ret = norm.pdf(x, mu, sigma) * prior
    # ret = multivariate_normal(mu, sigma).pdf(x) * prior
    print("value : ", ret, "mean : ", mu, "cov : ", sigma)
    return ret


def getRegion(x, mu, sigma, prior):
    r0 = region(x, mu[0], sigma[0], prior[0])
    r1 = region(x, mu[1], sigma[1], prior[1])

    # print("ro : ", r0)
    # print("r1 : ", r1)

    if r0 < r1:
        return 0
    else:
        return 1


def cdf_normal(z):
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))


def trueError(roots, prior, mu, sigma):
    roots = sorted(roots)
    r = -1
    cdfs = []
    for i in range(len(roots) + 1):
        if i == 0:
            r = getRegion(roots[i] - 0.1, mu, sigma, prior)
            print("f : ", r)
            if r == 0:
                z = (roots[i] - mu[0]) / np.sqrt(sigma[0])
                prior_prob = prior[0]
            else:
                z = (roots[i] - mu[1]) / np.sqrt(sigma[1])
                prior_prob = prior[1]

            cdf = cdf_normal(z) * prior_prob

        elif i == len(roots):
            if r == 1:
                z = (roots[i - 1] - mu[0]) / np.sqrt(sigma[0])
                prior_prob = prior[0]
                r = 0
            else:
                z = (roots[i - 1] - mu[1]) / np.sqrt(sigma[1])
                prior_prob = prior[1]
                r = 1

            cdf = (1 - cdf_normal(z)) * prior_prob

        else:
            if r == 0:
                z = (roots[i] - mu[1]) / np.sqrt(sigma[1])
                z_prev = (roots[i - 1] - mu[1]) / np.sqrt(sigma[1])
                prior_prob = prior[1]
                r = 1

            else:
                z = (roots[i] - mu[0]) / np.sqrt(sigma[0])
                z_prev = (roots[i - 1] - mu[0]) / np.sqrt(sigma[0])
                prior_prob = prior[0]
                r = 0

            cdf = (cdf_normal(z) - cdf_normal(z_prev)) * prior_prob

        # print("cdf : ", cdf)
        cdfs.append(cdf)

    return np.sum(cdfs)


number_points = [50, 100, 200, 500, 1000]
mu = [-0.5, 0.5]
all_sigma = [[1,1], [2,2], [2,2], [3,1]]
all_prior = [[0.5, 0.5], [2/3, 1/3], [0.5, 0.5], [0.5, 0.5]]

all_bhatt = []
all_true = []
all_emp = []

all_result = {}

q = ['7', '8(a)', '8(b)', '8(c)']
for i in range(len(all_prior)):

    sigma = all_sigma[i]
    prior = all_prior[i]
    all_bhatt = []
    all_true = []
    all_emp = []
    print("cov : ", sigma)

    print("-----------------------------------------------------------")
    print("distributions : "+"~(",mu[0], ", ", sigma[0], ") and ~(", mu[1], ", ", sigma[1], ") and prior : ", prior)
    print("-----------------------------------------------------------")

    for count in number_points:

        d1, d2 = generate(mu, sigma, count)
        predicted, actual, emperical_error = classify(d1, d2, mu, sigma, prior, count)
        bhatt_bound = bhattacharya_bound(mu[0], mu[1], sigma[0], sigma[1], prior[0], prior[1])

        roots = square_root(mu, sigma, prior)
        print("roots : ", roots)
        true_error = trueError(roots, prior, mu, sigma)
        print("emp errors with ", count, "datapoints : ", emperical_error)

        all_bhatt.append(bhatt_bound)
        all_emp.append(emperical_error)
        all_true.append(true_error)

    # plt.plot(all_bhatt, label = 'bhattacharya')
    # plt.plot(all_emp, label = "emperical error")
    # plt.plot(all_true, label = "true error")
    # plt.legend()
    # plt.title("error comparision graph")
    # plt.xlabel("number of data points")
    # plt.ylabel("errors")
    # plt.xticks(np.arange(len(number_points)), number_points)
    # plt.savefig('Q8/' + q[i] + '_.png')
    # plt.show()
    print("true errors :", np.unique(all_true))
    print("bhatt. bound :", np.unique(all_bhatt))
