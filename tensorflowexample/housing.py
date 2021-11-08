import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt",header=None,sep='\s+')
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

import matplotlib.pyplot as plt
import seaborn as sns
cols = ["LSTAT","INDUS","NOX","RM","MEDV"]
sns.pairplot(df[cols],height=2.5)
plt.tight_layout()
plt.show()
