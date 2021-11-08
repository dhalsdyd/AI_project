from numpy.lib.type_check import nan_to_num
import pandas as pd
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
#print(df_wine)

from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
#print(eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1,14),var_exp,alpha=0.5,align="center",label = "Individual explained variance")
plt.step(range(1,14),cum_var_exp,where='mid',label = 'Cumulateive explained variance')
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
print(eigen_pairs)
eigen_pairs.sort(key = lambda k : k[0],reverse = True)
print(eigen_pairs)

w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print(w)

X_train_pca = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train == l,0],X_train_pca[y_train == l,1],c=c,label =l,marker=m)
plt.xlabel('PC 1')
plt.ylabel("PC 2")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution = 0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max = X[:,0].min() -1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min() -1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.4,cmap = cmap)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],y=X[y==cl,1],alpha = 0.6,c = colors[idx],marker = markers[idx],label = cl,edgecolor = 'black')

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_PCA = pca.fit_transform(X_train_std)
X_test_PCA = pca.transform(X_test)
lr = LogisticRegression(solver='liblinear',multi_class="auto")
lr.fit(X_train_PCA,y_train)
plot_decision_regions(X_train_PCA,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel("PC 2")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y_train)
lr2 = LogisticRegression(solver ="liblinear",multi_class="auto")
lr2.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=lr2)
plt.xlabel('LD 1')
plt.ylabel("LD 2")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

from sklearn.datasets import make_moons
X,y = make_moons(n_samples=100,random_state=123)
print(X)
plt.scatter(X[y==0,0],X[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X[y==1,0],X[y==1,1],color='blue',marker='o',alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1],color='blue',marker='o',alpha=0.5)
ax[1].scatter(X_spca[y==0,0],np.zeros((50,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1].scatter(X_spca[y==1,0],np.zeros((50,1))-0.02,color='blue',marker='o',alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC 1")
plt.show()

from sklearn.decomposition import KernelPCA
scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)


plt.scatter(X_skernpca[y==0,0],X_skernpca[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],color='blue',marker='o',alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel("PC 2")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()

from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000,random_state=123,noise = 0.1,factor = 0.2)
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma = 15)
X_skernpca = scikit_kpca.fit_transform(X)

fig,ax = plt.subplots(nrows = 1,ncols = 3,figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1],color='red',marker='^',alpha=0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1],color='blue',marker='o',alpha=0.5)
ax[1].scatter(X_spca[y==0,0],np.zeros((500,1))+0.02,color='red',marker='^',alpha=0.5)
ax[1].scatter(X_spca[y==1,0],np.zeros((500,1))-0.02,color='blue',marker='o',alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel("PC 1")
plt.show()

plt.scatter(X_skernpca[y==0,0],X_skernpca[y==0,1],color='red',marker='^',alpha=0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],color='blue',marker='o',alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel("PC 2")
plt.legend(loc = "lower left")
plt.tight_layout()
plt.show()