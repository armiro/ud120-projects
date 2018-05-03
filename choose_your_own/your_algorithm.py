#!C:\Python27

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]


# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


# your code here!  name your classifier object clf if you want the
# visualization code (prettyPicture) to show you the decision boundary

from sklearn import ensemble
clf = ensemble.AdaBoostClassifier(n_estimators=40)

t0 = time()
clf.fit(features_train, labels_train)
print ('training time is: ', round(time()-t0, 3), 's')
t1 = time()
predictions = clf.predict(features_test)
print ('test time is: ', round(time()-t1, 3), 's')

from sklearn.metrics import accuracy_score
adaBoostAcc = accuracy_score(predictions, labels_test)
print ('The AdaBoost Accuracy is: ', adaBoostAcc)

# running the prettyPicture will create a .png image in the same directory
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

print ('----------------')

from sklearn import neighbors
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='uniform')

t2 = time()
clf_knn.fit(features_train, labels_train)
print ('training time is: ', round(time()-t2, 3), 's')
t3 = time()
predictions_knn = clf_knn.predict(features_test)
print ('test time is: ', round(time()-t3, 3), 's')

from sklearn.metrics import accuracy_score
knnAcc = accuracy_score(predictions_knn, labels_test)
print ('The kNN Accuracy is: ', knnAcc)

# try:
#     prettyPicture(clf_knn, features_test, labels_test)
# except NameError:
#     pass

print ('----------------')

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=25, min_samples_split=6, min_samples_leaf=2)

t4 = time()
clf_rf.fit(features_train, labels_train)
print ('training time is: ', round(time()-t4, 3), 's')
t5 = time()
predictions_rf = clf_rf.predict(features_test)
print ('test time is: ', round(time()-t5, 3), 's')

from sklearn.metrics import accuracy_score
rfAcc = accuracy_score(predictions_rf, labels_test)
print ('The RandomForest Accuracy is: ', rfAcc)

# try:
#     prettyPicture(clf_rf, features_test, labels_test)
# except NameError:
#     pass
