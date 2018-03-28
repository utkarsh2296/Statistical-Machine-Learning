import random
import numpy as np
from numpy.random.mtrand import seed
import matplotlib.pyplot as plt

mean = [2, 3]
sigma = np.array([[1, 1.5], [1.5, 30]])
seed(10)
data = np.random.multivariate_normal(mean, sigma, 100)


# print(np.corrcoef(np.power(data[:, 0], 2), np.power(data[:, 1], 2)))


def variation(sigma, mu=[2, 3], power=2):
    y_var = [random.randint(2, 100) for i in range(30)]
    # print(y_variation)
    y_var = sorted(y_var)
    # print(y_variation)
    multiple_corr = []

    for i in y_var:
        new_sigma = sigma[:]
        new_sigma[1][1] = i

        seed(1)
        new_data = np.random.multivariate_normal(mu, new_sigma, 100)
        multiple_corr.append((np.corrcoef(np.power(new_data[:, 0], power), np.power(new_data[:, 1], power)))[1][0])

    return y_var, multiple_corr


def plot(x, y, title, xlabel, ylabel, scatter="off"):
    if scatter == "off":
        plt.plot(x, y)
    else:
        plt.scatter(x, y)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# part 1
correlation = np.corrcoef(data[:, 0], data[:, 1])
print("correlation is : ", correlation)

# part 2
y_variation, multiple_correlation = variation(sigma=sigma, power=1)

# plot
plot(y_variation, multiple_correlation, xlabel="different values of y", ylabel="carrelation values",
     title="graph of correlation with change in variance of Y")

# part 4
squared_data = np.square(data)
print(squared_data[:, 1])
print(data)
correlation_square = np.corrcoef(squared_data[:, 0], squared_data[:, 1])
print("correlation when values of data are squared : ", correlation_square)
plt.scatter(data[:, 0], data[:, 1], c='blue')
plt.scatter(squared_data[:, 0], squared_data[:, 0], c='red')
plt.show()

# part 5
new_mean = [-2, 3]
data = np.random.multivariate_normal(new_mean, sigma, 100)
print(data)

'''for X and Y'''
y_variation, multiple_correlation = variation(sigma=sigma, mu=new_mean, power=1)
plt.scatter(y_variation, multiple_correlation)
plt.show()

'''for X^2 and Y^2'''
y_variation_square, multiple_correlation_square = variation(sigma=sigma, mu=new_mean)
print(multiple_correlation_square)
plt.scatter(y_variation_square, multiple_correlation_square)
plt.show()
