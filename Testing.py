#! usr/bin/python3.5
import numpy as np
import PCApackage as pac 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

Y=pac.PCA(X,k=2, F=False)[0]
plt.subplot(2,2,1)
plt.title('After Linear PCA')
y_pred = KMeans(n_clusters=3, random_state=170).fit_predict(Y.T)
plt.scatter(Y[0], Y[1], c=y_pred)

Y=pac.PPCA(X, 2)
plt.subplot(2,2,2)
plt.title('After PPCA')
y_pred = KMeans(n_clusters=3, random_state=170).fit_predict(Y.T)
plt.scatter(Y[0], Y[1], c=y_pred)

Y=pac.PPCA(X, M=2, variance=False)
plt.subplot(2,2,3)
plt.title('After PPCA, zero sigma')
y_pred = KMeans(n_clusters=3, random_state=170).fit_predict(Y.T)
plt.scatter(Y[0], Y[1], c=y_pred)

my_pca=PCA(n_components=2)
Y = my_pca.fit_transform(X.T).T
plt.subplot(2,2,4)
plt.title('After Sklearn PCA')
y_pred = KMeans(n_clusters=3, random_state=170).fit_predict(Y.T)
plt.scatter(Y[0], Y[1], c=y_pred)
plt.show()