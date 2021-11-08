from sklearn_example import X_combined_std, X_train, X_train_std,y_train,y_combined,plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressionGD(object):
    def __init__(self,eta = 0.05,n_iter = 100,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0,scale= 0.01,size = 1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output))))
            self.cost_.append(cost)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    def activation(self,z):
        return 1.0 / (1.0 + np.exp(-np.clip(z,-250,250)))
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,0)
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta = 0.05,n_iter = 1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_regions(X = X_train_01_subset,y=y_train_01_subset,classifier=lrgd)
plt.xlabel("Sepel Length[cm]")
plt.ylabel("Petal Length[cm]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear",multi_class="auto",C=100.0,random_state=1)
lr.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = lr,test_idx=range(105,150))
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

from sklearn.svm import SVC
svm = SVC(kernel = "linear",C = 1.0,random_state=1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx = range(105,150))
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()