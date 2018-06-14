#!/C:/Python27


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
predictions = clf.predict(features)
acc = accuracy_score(labels, predictions)
print 'The Accuracy of over-fitted tree is:', round(acc, 3)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
predictions_on_holdout = clf.predict(features_test)
acc_on_holdout = accuracy_score(labels_test, predictions_on_holdout)
print 'The Accuracy on hold-out test data is:', round(acc_on_holdout, 3)


parameters = {
    'max_depth': [None, 3, 5],
    'min_samples_split': [10, 15, 20],
    'min_impurity_decrease': [0.0, 0.1, 0.15],
    'min_samples_leaf': [1, 5, 10]
    }

classifier = tree.DecisionTreeClassifier()
classifier = GridSearchCV(classifier, parameters)
classifier.fit(features_train, labels_train)
validated_pred = classifier.predict(features_test)
validated_acc = accuracy_score(labels_test, validated_pred)
print 'The Accuracy of GridSearchCV tree is:', round(validated_acc, 3)
print 'And its parameters are:', classifier.best_params_
