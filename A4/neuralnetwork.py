import numpy as np

class NeuralNetwork:
    def __init__(self, layers, softmax_=False):  # [784, 100, 50, 1]
        self.weights = []
        self.biases = []
        self.layers = layers
        self.softmax_ = softmax_
        self.predict_prob = []
        np.random.seed(1)

        for i in range(0, len(layers) - 1):
            w = np.random.uniform(-1, 1, (layers[i], layers[i + 1]))
            b = np.random.uniform(-1, 1, (1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def linear_activation(self, dot):
        pass

    def relu(self, dot):
        # print(dot)
        # print(np.maximum(0, dot))
        return np.maximum(0, dot)

    def relu_der(self, dot):
        for i in range(0, len(dot)):
            for k in range(len(dot[i])):
                if dot[i][k] > 0:
                    dot[i][k] = 1
                else:
                    dot[i][k] = 0
        return dot

    def sigmoid(self, dot):
        # print(dot)
        x = 1.0 / (1.0 + np.exp(-dot))
        return x

    def sigmoid_derivative(self, dot):
        return dot * (1.0 - dot)

    def softmax(self, dot):
        x = np.exp(dot) / np.sum(np.exp(dot))
        return x

    def cost(self, x):
        c = 0.5 * np.square(x)
        return c

    def fit(self, x, y, epoch=8, lr=0.01):

        for k in range(epoch):
            print_error = 0
            for i in (range(x.shape[0])):
                num = np.random.randint(x.shape[0])
                z = []
                a = [x[num]]
                z.append(a[-1])
                delta_error = []

                '''forward'''
                for i in range(len(self.weights) - 1):  # weights = [784x100], [100x50], [50x1]
                    dot = np.dot(a[i], self.weights[i]) + self.biases[i]
                    a.append(self.relu(dot))  # RELU
                    # a.append(self.sigmoid(dot))

                '''output layer'''
                z_last = None
                if self.softmax_:
                    # print("softwam")
                    z_last = self.softmax(np.dot(a[-1], self.weights[-1]) + self.biases[-1])

                else:
                    # print("sigmoid")
                    dot_last = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
                    z_last = self.sigmoid(dot_last)

                # z_last = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
                a.append(z_last)

                '''backpropagation #error'''
                cost_der = y[num] - a[-1]
                # cost_der = self.derivative(a[-1]) * cost_der
                delta_error.append(cost_der)

                # print(delta_error[-1].shape)
                # print(self.weights[0].shape)
                for l in range(len(a) - 2, 0, -1):
                    # print(self.weights[l].shape)
                    delta_error.append(delta_error[-1].dot(self.weights[l].T) * self.relu_der(a[l]))
                delta_error.reverse()

                # update the weights
                for i in range(len(self.weights)):
                    self.weights[i] += np.dot(np.atleast_2d(a[i]).T, np.array(delta_error[i])) * lr
                    self.biases[i] += delta_error[i] * lr

                print_error += self.cost(cost_der)
            print_error = print_error/len(x)
            print_error = np.sum(np.array(print_error).flatten())
            print("error at epoch ", str(k), " : ", print_error)

        return

    def predict(self, x):
        self.predict_prob = []
        out = []
        for i in range(x.shape[0]):
            a = [x[i]]

            # forward
            for i in range(len(self.weights) - 1):  # weights = [784x100], [100x50], [50x1]
                dot = np.dot(a[i], self.weights[i]) + self.biases[i]
                a.append(self.relu(dot))  # RELU
            # a.append(self.sigmoid(np.dot(a[-1], self.weights[-1]) + self.biases[-1]))  # OUTPUT
            # z_last = self.softmax(np.dot(a[-1], self.weights[-1]) + self.biases[-1])
            # z_last = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
            z_last = None
            if self.softmax_:
                z_last = self.softmax(np.dot(a[-1], self.weights[-1]) + self.biases[-1])
            else:
                dot_last = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
                z_last = self.sigmoid(dot_last)

            a.append(z_last)
            self.predict_prob.append(a[-1])
            if self.softmax_:
                out_ind = np.argmax(a[-1])
                row = [0] * self.layers[-1]
                row[out_ind] = 1
                out.append(row)

            else:
                if a[-1] >= 0.5:
                    out.append(1)
                else:
                    out.append(0)
        return np.array(out)

    def accuracy(self, p, a):
        count = 0
        if self.softmax_:
            for i in range(len(p)):
                p_in = np.argmax(p[i, :])
                a_in = np.argmax(a[i, :])

                if p_in == a_in:
                    count += 1
        else:
            p = p.flatten()
            for i in range(len(p)):
                if a[i] == p[i]:
                    count += 1

        return (count * 100.0) / len(p)

    def accuracy2(self, p, a):
        count = 0

        for i in range(len(p)):
            p_in = np.argmax(p[i,:])
            a_in = np.argmax(a[i,:])

            if p_in == a_in:
                count += 1

        return (count * 100.0) / len(p)



