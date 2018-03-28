import numpy as np
from sklearn.externals import joblib

d = joblib.load('projection_matrix/lda_projection_2.pkl')

print(d.shape)

import matplotlib.pyplot as plt

for i in range(len(d)):
    temp = d[i][:1024]
    temp = temp.reshape((32, 32))
    print(np.shape(temp))
    plt.imshow(temp)
    plt.savefig('plots/D2/LDA/' + str(i) + '.png')
    plt.clf()


x = [0, 1]
y = [0,0, 1]
z = [0, 1, 1]
plt.plot(x,x)
plt.plot(y, z)
