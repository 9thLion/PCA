#! usr/bin/python3.5

import numpy as np
import PCApackage as pac
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, factor=.3, noise=.05)

#First i split the data in two, to separate the small and the big circle
small=[]
big=[]
for i in range(len(X)):
    if y[i] == 0:
        big.append(X[i])
    if y[i] == 1:
        small.append(X[i])
        
Small=np.array(small)
Big=np.array(big)
BS=np.array([Big,Small]).reshape(1000,2).T 

plt.figure(figsize=(12,7))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.subplot(3,3,1)
plt.title('before PCA')
plt.scatter(Big.T[0], Big.T[1], c='c')
plt.scatter(Small.T[0], Small.T[1], c='r') 

Y1cov=pac.MyFirstAlgorithm(BS)[0].T
plt.subplot(3,3,2)
plt.title('after linear PCA (Eigendecomposition)')

plt.scatter(Y1cov[:500].T[0], np.zeros((500,1)), c='c') #The output is split in half, the last 500 are from the big circle
plt.scatter(Y1cov[500:].T[0], np.zeros((500,1)), c='r') #while the first 500 are from the small circle

Y1svd=pac.PCA(BS)[0].T
print('Shape of output matrix:', Y1svd.shape)
plt.subplot(3,3,3)
plt.title('after linear PCA (SVD)')

plt.scatter(Y1svd[:500].T[0], np.zeros((500,1)), c='c') 
plt.scatter(Y1svd[500:].T[0], np.zeros((500,1)), c='r') 
'''
The two eigenvectors are orthogonal because the covariance matrix
is symmetric and since they represent the line of maximum variance
it makes sense that they have the same power because our sample is arranged in
a circle.
The PCA is programmed so that it may choose the eigenvector number
automatically and since their eigenvalues are so close, the dimensions don't 
change at all.
'''

Y2novar=pac.PPCA(BS, variance=False)
plt.subplot(3,3,4)
plt.title('after PPCA, sigma = 0')

plt.scatter(Y2novar.T[:500].T[0], np.zeros((500,1)), c='c', alpha=0.5)
plt.scatter(Y2novar.T[500:].T[0], np.zeros((500,1)), c='r', alpha=0.5)

Y2var=pac.PPCA(BS, variance=True)
plt.subplot(3,3,5)
plt.title('after PPCA sigma not 0')

plt.scatter(Y2var.T[:500].T[0], np.zeros((500,1)), c='c', alpha=0.5)
plt.scatter(Y2var.T[500:].T[0], np.zeros((500,1)), c='r', alpha=0.5)
'''
Finally we run Kernel pca with 3 possible functions
'''

Y3g=pac.KPCA(BS) #gaussian

plt.subplot(3,3,6)
plt.title('after gaussian kernel PCA')
plt.scatter(Y3g.T[:500].T[0], np.zeros((500,1)), c='c', alpha=0.5)
plt.scatter(Y3g.T[500:].T[0], np.zeros((500,1)), c='r', alpha=0.5)

Y3p=pac.KPCA(BS, mode='polynomial')

plt.subplot(3,3,7)
plt.title('after polynomial kernel PCA')
plt.scatter(Y3p.T[:500].T[0], np.zeros((500,1)), c='c', alpha=0.5)
plt.scatter(Y3p.T[500:].T[0], np.zeros((500,1)), c='r', alpha=0.5)


Y3ht=pac.KPCA(BS, mode='hyperbolic tangent')

plt.subplot(3,3,8)
plt.title('after htangent kernel PCA')
plt.scatter(Y3ht.T[:500].T[0], np.zeros((500,1)), c='c', alpha=0.5)
plt.scatter(Y3ht.T[500:].T[0], np.zeros((500,1)), c='r', alpha=0.5)

plt.show()