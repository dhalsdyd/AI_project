from numpy.core.numeric import _convolve_dispatcher
from sklearn_example import X_train, X_train_std,X_combined_std,y_train,X_test,y_combined,plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
tree = DecisionTreeClassifier(criterion='gini',max_depth= 4,random_state=1)
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train,X_test))
plot_decision_regions(X_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel("Sepel Length[cm]")
plt.ylabel("Petal Length[cm]")
plt.legend(loc = "upper left")
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

# dot_data = export_graphviz(tree,filled=True,rounded=True,class_names=['Setosa','Versicolor','Virginica'],feature_names=['petal length','petal width'],out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('tree.png')

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = "gini",n_estimators=25,random_state=1,n_jobs=2)
forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.xlabel("Sepel Length[cm]")
plt.ylabel("Petal Length[cm]")
plt.legend(loc = "upper left")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric="minkowski")
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=knn,test_idx=range(105,150))

plt.xlabel("Sepel Length[stand]")
plt.ylabel("Petal Length[stand]")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

