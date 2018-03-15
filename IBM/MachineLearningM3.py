"""

Importing Libraries
Import the following libraries:

- load_diabetes from sklearn.datasets
- LinearRegression from sklearn.linear_model
- matplotlib.pyplot (as plt)
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

#creating an diabetes_data object/instance
diabetes = load_diabetes()

#1 independent variable using BMI
diabetes_X = diabetes.data[:, None, 2]

#instance of the LinearRegression called LinReg
LinReg = LinearRegression()


"""
Using train_test_split function to return xtrain/test set and ytrain/test set
X = feature matrix = diabetes_X
y = target response vector = diabetes.target
test_size = 0.3
random state = 7

Values to create train/test set
"""

X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)

#fitting using LinReg model

LinReg.fit(X_trainset, y_trainset)
#performing RSME
print(np.mean((LinReg.predict(X_testset) - y_testset) ** 2) ** (0.5) )

#visualisation
plt.scatter(X_testset, y_testset, color='black')
plt.plot(X_testset, LinReg.predict(X_testset), color='blue', linewidth=3)