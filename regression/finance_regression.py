#!/C:/Python27

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the data-set
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatter plot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load(open("../final_project/final_project_dataset_modified.pkl", "r"))

### list the features you want to look at--first item in the 
### list will be the "target" feature
# the most relevant feature to the bonus seems to be 'shared_receipt_with_poi'
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
sort_keys = '../tools/python2_lesson06_keys.pkl'
target, features = targetFeatureSplit(data)

### training-testing split needed in regression, just like classification
from sklearn import model_selection
feature_train, feature_test, target_train, target_test = model_selection.train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
print 'the slope is:', reg.coef_
print 'the intercept is:', reg.intercept_
predictions = reg.predict(feature_test)
scoreOnTraining = reg.score(feature_train, target_train)
scoreOnTest = reg.score(feature_test, target_test)
print 'The r2 score (of training made line) on training set is:', scoreOnTraining
print 'The r2 score (of training made line) on test set is:', scoreOnTest

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


### draw the regression line, once it's coded
try:
    plt.plot(feature_test, predictions, color="red")
except NameError:
    pass

reg.fit(feature_test, target_test)
scoreOnTraining_2 = reg.score(feature_train, target_train)
scoreOnTest_2 = reg.score(feature_test, target_test)
print 'the slope of test made line is:', reg.coef_
print 'the intercept of test made line is:', reg.intercept_
print 'The r2 score (of test made line) on training set is:', scoreOnTraining_2
print 'The r2 score (of test made line) on test set is:', scoreOnTest_2

plt.plot(feature_train, reg.predict(feature_train), color="blue")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
