#! usr/bin/python3.5
import numpy as np
import PCApackage as pac 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import os

os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
os.system('gunzip GDS6248.soft.gz')
os.system('tail -n +141 GDS6248.soft > GDS6248.softer') #getting rid of the redundant lines
os.system('rm GDS6248.soft')
os.system('head -n -1 GDS6248.softer > GDS6248.soft') #one last redundant line
os.system('rm GDS6248.softer')

#In the following loop I'm keeping the float values while skipping the strings by setting the ValueError exception
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
Color = ['m' for x in range(3)] + ['c' for x in range(24)] + ['r' for x in range(24)] #Color scheme for the samples

plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

green = mpatches.Patch(color='m',label='baseline')
cyan = mpatches.Patch(color='c',label='normal diet')
red = mpatches.Patch(color='r',label='high fat diet')

Y=pac.PCA(X,k=2, F=False)[0]
plt.subplot(2,2,1)
plt.title('After Linear PCA')
plt.scatter(Y[0], Y[1], c=Color)
plt.legend(bbox_to_anchor=(1, -.1), loc=2, borderaxespad=0.,handles=[green, cyan, red], prop={'size':10})
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
