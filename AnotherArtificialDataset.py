import numpy as np
import PCApackage as pac
import matplotlib.pyplot as plt

mean = np.ones(10)
var = np.identity(10)
X = np.random.multivariate_normal(mean,var,10).T
Y,S=pac.MyFirstAlgorithm(X,k=1) 
X2=X[:,:5] #5 samples
Y2,S2=pac.MyFirstAlgorithm(X2,k=1)
print(len(S))
print(np.zeros(Y.shape[0]).shape)
plt.figure(figsize=(6,8))
plt.subplot(2,3,1)
plt.title('Data Distribution')
plt.hist(X.T)
plt.subplot(2,3,2)
plt.title('Eigenvalues, 1000 samples')
plt.scatter(list(range(len(S))),S)
plt.subplot(2,3,3)
plt.title('After PCA, 1000 samples')
plt.scatter(Y[0],np.zeros(Y.shape[1]), c='c')
plt.subplot(2,3,4)
plt.title('Data Distribution')
plt.hist(X2.T)
plt.subplot(2,3,5)
plt.title('Eigenvalues, 500 samples')
plt.scatter(list(range(len(S2))),S2)
plt.subplot(2,3,6)
plt.title('After PCA, 500 samples')
plt.scatter(Y2[0],np.zeros(Y2.shape[1]), c='c')
plt.show()