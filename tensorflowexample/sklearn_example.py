from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("클래스 레이블 : ",np.unique(y))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(max_iter=40,eta0 = 0.1,tol = 1e-3,random_state=1)
ppn.fit(X_train_std,y_train)

from sklearn.metrics import accuracy_score
y_pred = ppn.predict(X_test_std)
print("정확도 : %.2f" % accuracy_score(y_test,y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X,y,classifier,test_idx = None,resolution = 0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max = X[:,0].min() -1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min() -1,X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.3,cmap = cmap)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl,0],y=X[y==cl,1],alpha = 0.8,c = colors[idx],marker = markers[idx],label = cl,edgecolor = 'black')
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c='1',edgecolors='black',alpha=0.2,linewidths=1,marker='o',s=100,label='testset')

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.show()




# 위와 같은 방법으로 선형적으로 구분되지 않을 때 수렴할 수 없다는 단점이 있다. -> 로지스틱 회귀로 가즈아!

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes,train_scores,test_scores = learning_curve(estimator = ppn,X=X_train_std,y = y_train,train_sizes= np.linspace(0.1,1.0,10),cv = 10,n_jobs = 1)
train_mean = 

