import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0700','#0900FF', '#00FF42'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(X_train[0])
# print(y_train)

# plt.figure()
# plt.scatter(X[:,0],X[:,1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

from knn import KNN
clf = KNN(k=4)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)


plt.figure()
plt.scatter(X_test[:,0],X_test[:,1], c=predictions, cmap=cmap, edgecolor='k', s=20)
plt.title("KNN for all the test datapoints")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# points where it failed
plt.figure()
plt.scatter(X_test[np.argwhere(predictions!=y_test),0], X_test[np.argwhere(predictions!=y_test),1],c="red", cmap=cmap, edgecolor='k', s=20)
plt.title("Points where the classifier failed")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()