import matplotlib.pyplot as plt
from sklearn_example import X_train_std,y_train,plot_decision_regions,X_combined_std,y_combined
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor == 1,0],X_xor[y_xor ==1,1],c = 'b',marker='x',label = '1')
plt.scatter(X_xor[y_xor == -1,0],X_xor[y_xor ==-1,1],c = 'r',marker='s',label = '-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc = "best")
plt.tight_layout()
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel = "rbf",C = 10.0,random_state=1,gamma=0.1)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier=svm)
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

svm2 = SVC(kernel = "rbf",C = 1.0,random_state=1,gamma = 0.2)
svm2.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=svm2,test_idx = range(105,150))
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()