from numpy.lib.shape_base import column_stack
from perceptron import X,y,plot_decision_regions
import numpy as np
class AdalineGD(object):
    def __init__(self,eta = 0.1,n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0,scale = 0.01,size = 1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        return self
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    def activation(self,X):
        return X
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0,1,-1)

import matplotlib.pyplot as plt

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15,eta = 0.01)
ada.fit(X_std,y)

# plot_decision_regions(X_std,y,classifier = ada)
# plt.title("Adaline - Gradient Descent")
# plt.xlabel("Sepel Length[standardized]")
# plt.ylabel("Petal Length[standardized]")
# plt.legend(loc = "upper left")
# plt.tight_layout()

# plt.show()

# plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker = 'o')
# plt.xlabel("Epochs")
# plt.ylabel("Sum-Squared-Error")
# plt.show()

class AdalineSGD(object):
    def __init__(self,eta = 0.01,n_iter = 10,shuffle = True,randome_state = None):
        self.eta =eta
        self.n_iter =n_iter
        self.w_initialized = False
        self.shuffle= shuffle
        self.random_state = randome_state
    def fit(self,X,y):
        self._initialized_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X,y = self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    def partial_fit(self,X,y):
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r],y[r]
    def _initialized_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0,scale = 0.01,size = 1 + m)
        self.w_initialized = True
    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    def activation(self,X):
        return X
    def predict(self,X):
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)

ada2 = AdalineSGD(n_iter = 15,eta = 0.01,randome_state= 1)
ada2.fit(X_std,y)

plot_decision_regions(X_std,y,classifier = ada2)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel("Sepel Length[standardized]")
plt.ylabel("Petal Length[standardized]")
plt.legend(loc = "upper left")
plt.tight_layout()

plt.show()

plt.plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker = 'o')
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared-Error")
plt.show()