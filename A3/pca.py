import numpy as np
import os
import cv2


class PCA:
    def __init__(self):
        self.eigen_values = []
        self.eigen_vectors = []

    def calculateEigens(self, X):
        # mean = np.mean(X, axis=0)
        # x_std = np.subtract(X, mean)
        # cov = np.matrix.dot(x_std.T, x_std)
        # cov = (1/(len(x_std))) * cov
        cov = np.cov(X.T)
        print("cov done")
        print(cov.shape)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(cov)
        print("eigenvector done!!")
        self.eigen_vectors = self.eigen_vectors.T

    def fit(self, X, eigen_energy=0.99):
        print("calculating eigen vectors......")
        self.calculateEigens(X)
        eigen_values_sum = np.sum(self.eigen_values)

        eigen_energy_vectors = []
        sum_count = 0

        for i in range(len(self.eigen_values)-1, 0, -1):
            # print("sum before : ", sum_count)
            sum_count += self.eigen_values[i]
            # print("sum after : ", sum_count)
            # print(float(sum_count/eigen_values_sum))

            if eigen_energy >= float(sum_count / eigen_values_sum):
                eigen_energy_vectors.append(self.eigen_vectors[i])
                continue
            else:
                break
        eigen_energy_vectors = np.array(eigen_energy_vectors)
        self.eigen_vectors = eigen_energy_vectors
        print("eigen_vector shape : ", self.eigen_vectors.shape)
        return

    def transform(self, X):
        return np.dot(X, self.eigen_vectors.T)



