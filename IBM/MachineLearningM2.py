"""
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics



#Using my_data as the skulls.csv data read by pandas, declare the following variables:

#X as the Feature Matrix (data of my_data)
#y as the response vector (target)
#targetNames as the response vector names (target names)
#featureNames as the feature matrix column names


my_data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")


# setting y as the response vector (target)
def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)


# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values


# setting x + y data using the remove column 1 coloumn function of my data,
X = removeColumns(my_data, 0, 1)
y = target(my_data, 1)

# train_test_split data returns 4 parameters, xtrain+ xtest, ytrain + ytest, the ratio (0.3) to train/test and random to ensure same splits
X_trainSet, X_testSet, y_trainSet, y_testSet = train_test_split(X, y, test_size=0.3, random_state=7)

# printing values to check whether the two sets match up
print(X_trainSet.shape)  # returns (105,4)
print(y_trainSet.shape)  # returns (105, )

print(X_testSet.shape)  # returns (45,4)
print(y_testSet.shape)  # returns (45,)

# three declarations of Kneighbours classifier and test different clusteringof values, 1, 23, 90
neigh = KNeighborsClassifier(n_neighbors=1)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh90 = KNeighborsClassifier(n_neighbors=90)

# fitting the instances with the Kneighbour classifier
neigh.fit(X_trainSet, y_trainSet)
neigh23.fit(X_trainSet, y_trainSet)
neigh90.fit(X_trainSet, y_trainSet)

# predicting Y value using x_testset
pred = neigh.predict(X_testSet)
pred23 = neigh.predict(X_testSet)
pred90 = neigh.predict(X_testSet)

# prediction accuracy of neigh,neigh23,neigh90 using metric.accuracy_score function
print("Neigh's Accuracy: ", metrics.accuracy_score(y_testSet, pred))
print("Neigh23's Accuracy: ", metrics.accuracy_score(y_testSet, pred23))
print("Neigh90's Accuracy: ", metrics.accuracy_score(y_testSet, pred90))


pl.scatter(X[:,0],X[:,1], y)
pl.xlabel ("Xtestdata")
pl.ylabel ("Ytest data")
pl.show()
"""
"""
#trying diabetes

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

#instance of diabetes from sk.datasets
diabetes = load_diabetes()
#setting the data of diabetes
diabetes_X = diabetes.data[:, None, 2]
#creating linear regression model
linReg = LinearRegression()

#setting up/training/testing diabetes independent values (X) (feature_matrix), and dependent values (target) (response vector)
X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)

#Train the LinReg model using X_trainset and y_trainset
linReg.fit(X_trainset,y_trainset)

#plotting using scatter plot
plt.scatter(X_testset, y_testset, color='black')
plt.plot(X_testset, linReg.predict(X_testset), color='blue', linewidth=3)
plt.show()
"""

#decision tree

import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from pydotplus import graphviz
from matplotlib import pyplot as plt
from sklearn.externals.six import StringIO
import matplotlib.image as mpimg
from sklearn import tree

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

#
#X as the Feature Matrix (data of my_data)
#y as the response vector (target)
#targetNames as the response vector names (target names)

targetNames = my_data["epoch"].unique().tolist()
targetNames
y = my_data["epoch"]
y[0:5]
#featureNames as the feature matrix column names
featureNames = list(my_data.columns.values)[2:6]
# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0,1]], axis=1).values
X[0:5]

# train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size = 0.3, random_state = 7)

#Print the shape of X_trainset and y_trainset. Ensure that the dimensions match
print ("X train:",X_trainset.shape, "y train:",y_trainset.shape)

#Print the shape of X_testset and y_testset. Ensure that the dimensions match
print ("X test:",X_testset.shape, "Y test:",y_testset.shape)

# create an instance of the DecisionTreeClassifier called skullsTree + entropy value.
skullsTree = DecisionTreeClassifier(criterion="entropy")
#fit the data with the training feature matrix X_trainset and training response vector y_trainset
skullsTree.fit(X_trainset,y_trainset)

#predictions on the testing dataset and store it into a variable called predTree.
predTree = skullsTree.predict(X_testset)

#print out predTree and y_testset to visually compare the prediction to the actual values.
print ("prediction values:", predTree [0:5],"Actual values:", y_testset [0:5])

#check the accuracy of our model.
print("DecisionTrees's Accuracy: ", accuracy_score(y_testset, predTree))

"""
#visualising data - Not working, why?
dot_data = StringIO()
filename = "skulltree.png"
out=tree.export_graphviz(skullsTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
"""

#Random Forests

from sklearn.ensemble import RandomForestClassifier

#Create an instance of the RandomForestClassifier() called skullsForest, where the forest has 10 decision tree estimators (n_estimators=10) and the criterion is entropy (criterion="entropy")
skullsForest = RandomForestClassifier(n_estimators= 10, criterion = "entropy")
#fitting skull forest with decision tree data
skullsForest.fit(X_trainset, y_trainset)
#predicting using X_testset variable
predForest = skullsForest.predict(X_testset)
print ("prediction values:", predForest,"Actual values:", y_testset)
#checking accuracy of model:
print("RandomForests's Accuracy: ",accuracy_score(y_testset, predForest))

#trees are in our skullsForest variable by using the .estimators_ attribute. This attribute is indexable, so we can look at any individual tree we want.

print(skullsForest.estimators_)

#visuals
#choose to view any tree by using the code below. Replace the "&" in skullsForest[&] with the tree you want to see.

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
#Replace the '&' below with the tree number
tree.export_graphviz(skullsForest,
                     out_file=dot_data,
                     feature_names=featureNames,
                     class_names=targetNames,
                     filled=True, rounded=True,
                     special_characters=True,
                     leaves_parallel=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
