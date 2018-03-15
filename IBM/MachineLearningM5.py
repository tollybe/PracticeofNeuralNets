##Collaborative Filtering
"""
data = {"Jacob": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 3.5, "Edge of Tomorrow": 5.0, "The Maze Runner": 3.0},
     "Jane": {"The Avengers": 3.0, "The Martin": 4.0, "Guardians of the Galaxy": 4.0, "Edge of Tomorrow": 3.0, "The Maze Runner": 3.0, "Unbroken": 4.5},
     "Jim": {"The Martin": 1.0, "Guardians of the Galaxy": 2.0, "Edge of Tomorrow": 5.0, "The Maze Runner": 4.5, "Unbroken": 2.0}}

#example of movies watched by Jacob
data.get("Jacob")
common_movies = list(set(data.get("Jacob")).intersection(data.get("Jane")))

#find recommendations using difference function
recommendation = list(set(data.get("Jane")).difference(data.get("Jacob")))

print (common_movies)
print (recommendation)

#using similar means function to find mean difference in rating
def similar_mean(same_movies, user1, user2, dataset):
    total = 0
    for movie in same_movies:
        total += abs(dataset.get(user1).get(movie) - dataset.get(user2).get(movie))


print(similar_mean(common_movies, "Jacob", "Jane", data))
#find movie from Jacob for Jim
common_movies2 = list(set(data.get("Jacob")).intersection(data.get("Jim")))
"""
"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas

sel = VarianceThreshold()
#dataset
dataset = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]

sel.fit_transform(dataset)
#instatiated object that has 60% threshold
sel60 = VarianceThreshold(threshold=(0.6 * (1 - 0.6)))
#transforms dataset
sel60.fit_transform(dataset)

my_data = pandas.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv", delimiter=",")

#function to remove column
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

#remove column [0,1] from dataset
X = removeColumns(my_data, 0, 1)

#creates target function to obtain response vector and store as y
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

y = target(my_data, 1)
X.shape
#uses fit transform_function for X,y with parameters, k3 + chi2
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
X_new.shape

##univariance ##

dataset = [
     {'Day': 'Monday', 'Temperature': 18},
     {'Day': 'Tuesday', 'Temperature': 13},
     {'Day': 'Wednesday', 'Temperature': 7}
      ]
#instance of dictvectoriser
vec = DictVectorizer()

#transform data
vec.fit_transform(dataset).toarray()

#select featurefunction to review
vec.get_feature_names()

#import plot chart things
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0)
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], c=y, cmap=plt.cm.spectral)

#3d representation of data
pca = decomposition.PCA(n_components=2)

#fit/transform data
pca.fit(X_new)
PCA_X = pca.transform(X_new)

fig = plt.figure(1, figsize=(10, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=0)
ax.scatter(PCA_X[:, 0], PCA_X[:, 1], c=y, cmap=plt.cm.spectral)

PCA_X.shape
"""
###SUPPORT VECTOR REGRESSION ###

from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


np.random.seed(5)
X = np.sort(10 * np.random.rand(30, 1), axis=0)
y = np.sin(X).ravel()

#three dif. kernels rbf, linear and sigmoid
svr_rbf = SVR(kernel='rbf', C=1e3)

svr_linear = SVR(kernel='linear', C=1e3)
svr_sigmoid = SVR(kernel='sigmoid', C=1e3)

#fit function

svr_rbf.fit(X,y)
svr_linear.fit(X,y)
svr_sigmoid.fit(X,y)
SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#predict function
y_pred_rbf = svr_rbf.predict(X)
y_pred_linear = svr_linear.predict(X)
y_pred_sigmoid = svr_sigmoid.predict(X)

#create scatterplot

plt.scatter(X, y, c='k', label='data')
plt.plot(X, y_pred_rbf, c='g', label='RBF model')
plt.plot(X, y_pred_linear, c='r', label='Linear model')
plt.plot(X, y_pred_sigmoid, c='b', label='Sigmoid model')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
