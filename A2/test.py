import numpy as np
import math

from scipy.stats import norm

mu = [-0.5, 0.5]
sigma = [2, 2]
prior = [2/3, 1/3]


def square_root(mu, sigma, priors):
    A = 0.5 * float((1 / sigma[0]) - float(1 / sigma[1]))
    B = -(float((mu[0]) / sigma[0])) + float((mu[1]) / sigma[1])
    C = 0.5 * (float(np.square(mu[0]) / sigma[0]) - float(np.square(mu[1]) / sigma[1])) - 0.5 * float(
        np.log(sigma[1] / sigma[0])) + float(np.log(
        priors[1] / priors[0]))
    p = [A, B, C]
    print("A : ", A)
    print("B : ", B)
    print("C : ", C)
    roots = np.roots(p)
    print("roots : ", roots)
    return roots


def region(x, mu, sigma, prior):
    return norm.pdf(x, mu, sigma) * prior


def getRegion(x, mu, sigma, prior):
    r0 = region(x, mu[0], sigma[0], prior[0])
    r1 = region(x, mu[1], sigma[1], prior[1])

    if r0 < r1:
        return 0
    else:
        return 1


def cdf_normal(z):
    return 0.5 * (1 + math.erf(z / np.sqrt(2)))


roots = square_root(mu, sigma, prior)
cdfs = []

# if roots.__len__() == 1:
#     r = getRegion(roots[0], mu, sigma, prior)
#     if r==0:
#         #limit will be -infinity to x
#
# else:
roots = sorted(roots)
r = -1
for i in range(len(roots) + 1):
    if i == 0:
        r = getRegion(roots[i] - 5, mu, sigma, prior)
        print("f : ", r)
        if r == 0:
            z = (roots[i] - mu[0]) / np.sqrt(sigma[0])
            prior_prob = prior[0]
        else:
            z = (roots[i] - mu[1]) / np.sqrt(sigma[1])
            prior_prob = prior[1]

        cdf = cdf_normal(z)*prior_prob

    elif i == len(roots):
        if r == 1:
            z = (roots[i-1] - mu[0]) / np.sqrt(sigma[0])
            prior_prob = prior[0]
            r = 0
        else:
            z = (roots[i-1] - mu[1]) / np.sqrt(sigma[1])
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

    print("cdf : ", cdf)
    cdfs.append(cdf)


'''Question 8'''
'''part (a)'''
mu = [-0.5, 0.5]
sigma = [2, 2]
prior = [2/3, 1/3]
