"""
Ex. 1
#imports digit
from sklearn.datasets import load_digits
#imports algorithm
from sklearn import svm

digits = load_digits()

##type of digit - stored as a numpy datatype which is a ndarray
print (type(digits))

#printing a list of data
print (digits.data)
x = digits.data
#printing description of the data
#print (digits.DESCR)
#calling the target field to fetch numbers where each digit is mapped to a name in target_names
print (digits.target)
y = digits.target
#print target_names
print(digits.target_names)
#shapes the data
print (digits.data.shape)
print (digits.target.shape)

# calling on svm - declare a variable called clf with gamma and C attributes.
clf = svm.SVC(gamma = 0.001,C = 100)
clf.fit(x,y)
print("Prediction: ", clf.predict(digits.data[-1]))
print ("Actual: ", y[-1])
"""
"""
Ex 2

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

path = pandas.read_csv("https://ibm.box.com/shared/static/u8orgfc65zmoo3i0gpt9l27un4o0cuvn.csv", delimiter=",")

#print (path) #prints files
#print(type(path)) # prince type being pandas df

values = path.values
shape = path.shape
columnHeaders = path.columns

print(values,columnHeaders,shape) #printing values, column titles and datatype

# function removes epoch categorisation column titles
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]],axis = 1).values

newPath = removeColumns(path,0,1)
print(newPath)

#creares response vector y
def targetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(path.values)):
        if path.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[path.values[i][targetColumnIndex]] = count
        target.append(target_dict[path.values[i][targetColumnIndex]])
    # Since a dictionary is not ordered, we need to order it and output it to a list so the
    # target names will match the target.
    for targetName in sorted(target_dict, key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names

target, target_names = targetAndtargetNames(path, 1)

print( target, target_names)

X = newPath
y = target
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
print('Prediction: '), neigh.predict(newPath[10])
print('Actual:'), y[10]

"""


"""
Ex 3
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#calling X value
my_data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

#remove column
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]],axis =1).values

#setting target value (y)
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

X = removeColumns(my_data,0,1)
y = target(my_data,1)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
neigh7 = KNeighborsClassifier(n_neighbors=7)
neigh7.fit(X,y)

print("Neigh prediction: ", neigh.predict(X[30]))
print("Neigh7 prediction: ", neigh7.predict(X[30]))
print( "actual: ", y[30])