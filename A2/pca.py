import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class PCA:
    def __init__(self):
        self.eigen_values = []
        self.eigen_vectors = []

    def read(self, path):
        print("calling for ", path)
        data = []
        label = []
        shape = []

        for file in os.listdir(path):
            if not os.path.isfile(path + '/' + file):
                ret_data, ret_label, ret_shape = self.read(path + '/' + file)
                data.extend(ret_data)
                label.extend(ret_label)
                shape.extend(ret_shape)
            else:
                img = cv2.imread(path + '/' + file, -1)
                img = cv2.resize(img, (50, 50))
                # print((img.shape))
                shape.append(np.array(img).shape)
                img = img.flatten()

                data.append(img)
                label.append(int(os.path.basename(path)))
        return np.array(data), np.array(label), np.array(shape)

    def calculateEigens(self, X):
        mean = np.mean(X, axis=0)
        x_std = np.subtract(X, mean)
        cov = np.matrix.dot(x_std.T, x_std)
        cov = (1/(len(x_std))) * cov
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(cov)
        self.eigen_vectors = self.eigen_vectors.T

    def getEigenVectors(self, X, eigen_energy=0.99):
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
        return eigen_energy_vectors

    def transform(self, X, eigen_vectors):
        return np.dot(X, eigen_vectors.T)

    def plot(self, eigen_vectors):
        count = 0
        f, axarr = plt.subplots(np.ceil(len(eigen_vectors)/4), 4)

        index = 0
        for row in np.ceil(len(eigen_vectors)/4):
            flag = 0
            for col in range(4):
                axarr[row,col].imshow(eigen_vectors[index])
                index += 1
                if index >= len(eigen_vectors):
                    flag = 1
                    break
            if flag == 1:
                break

        # for i in eigen_vectors:
        #
        #     v = i
        #     img = v.reshape((50, 50))
        #     plt.imshow(img, cmap='gray')
        #     plt.savefig(str(3) + '/' + str(count) + '.png')
        #     count += 1
