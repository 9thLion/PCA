#! usr/bin/python3.5
import numpy as np
import PCApackage as pac 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
os.system('gunzip GDS6248.soft.gz')
os.system('tail -n +141 GDS6248.soft > GDS6248.softer') #getting rid of the redundant lines
os.system('rm GDS6248.soft')
os.system('head -n -1 GDS6248.softer > GDS6248.soft') 
os.system('rm GDS6248.softer')

temp = []
with open('GDS6248.soft') as f:
	for l in f:
		temp2=[]
		for x in l.split()[2:]:
			try:
				temp2.append(float(x))
			except ValueError: 
				pass
		temp.append(temp2)

X=np.array(temp)
Color = ['w' for x in range(3)] + ['c' for x in range(24)] + ['r' for x in range(24)]

plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


Y=pac.PCA(X,k=2, F=False)[0]
plt.subplot(2,2,1)
plt.title('After Linear PCA')
plt.scatter(Y[0], Y[1], c=Color)

Y=pac.PPCA(X, 2)
plt.subplot(2,2,2)
plt.title('After PPCA')
plt.scatter(Y[0], Y[1], c=Color)


Y=pac.PPCA(X, M=2, variance=False)
plt.subplot(2,2,3)
plt.title('After PPCA, zero sigma')
plt.scatter(Y[0], Y[1], c=Color)

my_pca=PCA(n_components=2)
Y = my_pca.fit_transform(X.T).T
plt.subplot(2,2,4)
plt.title('After Sklearn PCA')
plt.scatter(Y[0], Y[1], c=Color)
plt.show()
