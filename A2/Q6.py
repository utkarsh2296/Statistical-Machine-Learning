import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from pca import PCA

pca = PCA()
X, Y, _ = pca.read('Face_data')
eigen_energy_vectors = {90: pca.getEigenVectors(X, eigen_energy=0.90), 95: pca.getEigenVectors(X, eigen_energy=0.95),
                        99: pca.getEigenVectors(X, eigen_energy=0.99)}

train_x, test_x, train_y, test_y = train_test_split(X, Y, shuffle=True, test_size=0.3, stratify=Y, random_state=42)
svm = LinearSVC()
svm.fit(train_x, train_y)
predicted = svm.predict(test_x)
print("accuracy on actual data: ", svm.score(test_x, test_y))

for key in eigen_energy_vectors.keys():
    transform_X = pca.transform(X=X, eigen_vectors=eigen_energy_vectors[key])
    # joblib.dump(eigen_energy_vectors[key], 'Q6/' + str(key) + '_projection_matrix.pkl')
    train_x, test_x, train_y, test_y = train_test_split(transform_X, Y, shuffle=True, test_size=0.3, stratify=Y,
                                                        random_state=42)
    svm = LinearSVC()
    svm.fit(train_x, train_y)
    print("accuracy on transformed data at " + str(key) + '% eigen energy : ', svm.score(test_x, test_y))
