import numpy as np


def inverse(matrix):
    u, s, v = np.linalg.svd(matrix)
    matrix_inv = np.dot (v.transpose(), np.dot(np.diag(s**-1),u.transpose()))
    return(matrix_inv)


class LDA:
    def __init__(self):
        self.eigen_values = []
        self.eigen_vectors = []
        self.means = {}
        self.complete_mean = []
        self.within_class_scatter = []
        self.between_class_scatter = []

    def calculate_mean(self, X, Y):
        classes = np.unique(Y)
        Y = Y.reshape((len(Y), 1))

        for c in classes:
            mean = np.mean(X[np.where(Y[:, 0] == c)], axis=0)
            self.means[c] = mean

        self.complete_mean = np.mean(X, axis=0)

    def within_class_scatter_matrix(self, X, Y):
        within_class_scatter = np.zeros((X.shape[1], X.shape[1]))
        classes = np.unique(Y)
        Y = Y.reshape((len(Y), 1))

        for c in classes:
            data = X[np.where(Y[:, 0] == c)]
            scatter = np.cov(data.T, bias=1) * (data.shape[0] - 1)
            # scatter = np.cov(data.T, bias=1)
            # print("scatter shape : ", scatter.shape)
            within_class_scatter = np.add(within_class_scatter, scatter)
            # print("rank : ", np.linalg.matrix_rank(within_class_scatter))

        self.within_class_scatter = within_class_scatter

    def between_class_scatter_matrix(self, X, Y):
        between_scatter = np.zeros((len(self.complete_mean), len(self.complete_mean)))
        classes = np.unique(Y)
        Y = Y.reshape((len(Y), 1))

        for c in classes:
            mean = self.means[c]
            data = X[np.where(Y[:, 0] == c)]
            length = data.shape[0]
            mean = np.subtract(mean, self.complete_mean)
            # print("mean : ", mean)
            mean = mean.reshape((1, len(mean)))

            between_scatter = np.add(np.dot(mean.T, mean) * length, between_scatter)

        self.between_class_scatter = between_scatter

    def fit(self, X, Y):
        n_eigen = len(np.unique(Y)) - 1
        self.calculate_mean(X, Y)
        self.within_class_scatter_matrix(X, Y)
        self.between_class_scatter_matrix(X, Y)
        # matrix = np.dot(np.invert(self.within_class_scatter.astype(int)), self.between_class_scatter)
        matrix = np.dot(inverse(self.within_class_scatter), self.between_class_scatter)
        # self.eigen_values, self.eigen_vectors = eigsh(matrix, k=n_eigen)
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(matrix)
        self.eigen_vectors = self.eigen_vectors.T
        self.eigen_vectors = self.eigen_vectors[-n_eigen:]
        print("sjdchdhghjd")
        print("eigen : ", self.eigen_vectors.shape)
        return

    def transform(self, X):
        # transform_X = []
        # for i in X:
        #     transform_X.append(np.matmul(i, self.eigen_vectors.T))
        # return transform_X
        return np.matmul(X, self.eigen_vectors.T)

    # def plot(self, eigen_vectors):
    #     count = 0
    #     f, axarr = plt.subplots(np.ceil(len(eigen_vectors)/4), 4)
    #
    #     index = 0
    #     for row in np.ceil(len(eigen_vectors)/4):
    #         flag = 0
    #         for col in range(4):
    #             axarr[row,col].imshow(eigen_vectors[index])
    #             index += 1
    #             if index >= len(eigen_vectors):
    #                 flag = 1
    #                 break
    #         if flag == 1:
    #             break
    #
    #     # for i in eigen_vectors:
    #     #
    #     #     v = i
    #     #     img = v.reshape((50, 50))
    #     #     plt.imshow(img, cmap='gray')
    #     #     plt.savefig(str(3) + '/' + str(count) + '.png')
    #     #     count += 1

# lda = LDA()
# X, Y, _ = lda.read('Face_data')
# print(X.shape)
#
# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, shuffle=True)
# pca = PCA()
# pca_eigen_vector = pca.getEigenVectors(train_x)
# pca_transform_x = pca.transform(train_x, pca_eigen_vector)
# lda.fit(pca_transform_x, train_y)
#
# # print("eigen values : ", lda.eigen_values)
#
# '''classifier'''
# transformed_train_data = lda.transform(pca_transform_x)
# print(transformed_train_data.shape)
# pca_transform_test_x = pca.transform(test_x, pca_eigen_vector)
# transformed_test_data = lda.transform(pca_transform_test_x)
#
# lr = LogisticRegression()
# lr.fit(transformed_train_data, train_y)
# print(lr.score(transformed_test_data, test_y))
#
# # c = 0
# # for t in lda.eigen_vectors:
# #
# #     t = np.array(t).reshape((50, 50))
# #     plt.imshow(t)
# # # plt.plot(lda.eigen_vectors[0])
# # #     plt.show()
# #     c += 1
# #     plt.savefig("rough/" + str(c) + ".png")
# #     plt.clf()
#
# print(lda.eigen_values[-10:])
