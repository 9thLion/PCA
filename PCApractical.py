#! usr/bin/python3.5
import numpy as np
import PCApackage as pac 
import matplotlib.pyplot as plt
from sklearn import decomposition
import os

os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
os.system('gunzip GDS6248.soft.gz')
os.system('tail -n +141 GDS6248.soft > GDS6248.softer') #getting rid of the redundant lines
os.system('rm GDS6248.soft')
os.system('head -n -1 GDS6248.softer > GDS6248.soft') 

with open('GDS6248.soft') as f:
	testsite_array = f.readlines()
#! usr/bin/python3.5
import numpy as np
import PCApackage as pac 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

temp = []
#lengths=[]
with open('GDS6248.soft') as f:
	for l in f:
		temp2=[]
		for x in l.split()[2:]:
			try:
				temp2.append(float(x))
			except ValueError: 
				#there are some redundant words in columns 3 and 4 sometimes
				pass
#		lengths.append(len(temp2))
		temp.append(temp2)
X=np.array(temp)
print(X.shape)

Y=pac.PPCA(X)
print(Y.shape)
plt.subplot(2,2,1)
plt.scatter(Y[0], np.zeros((51,1)), c='c', alpha=0.5)
Y=pac.PPCA(X, variance=False)
plt.subplot(2,2,2)
plt.scatter(Y[0], np.zeros((51,1)), c='c', alpha=0.5)

my_pca=PCA(n_components=2)
Y = my_pca.fit_transform(X.T).T
plt.subplot(2,2,3)
plt.scatter(Y[0], Y[1], c='c', alpha=0.5)

plt.show()