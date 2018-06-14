#!/C:/Python27


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score

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

numOfPOIs = 0
for label in labels_test:
    if label == 1.0:
        numOfPOIs += 1
print 'Number of POIs in the test-set is:', numOfPOIs
print 'Total number of people in the test-set is:', len(labels_test)

print 'The Precision score is:', precision_score(labels_test, predictions_on_holdout)
print 'The Recall score is:', recall_score(labels_test, predictions_on_holdout)

# parameters = {
#     'max_depth': [None, 3, 5],
#     'min_samples_split': [10, 15, 20],
#     'min_impurity_decrease': [0.0, 0.1, 0.15],
#     'min_samples_leaf': [1, 5, 10]
#     }
#
# classifier = tree.DecisionTreeClassifier()
# classifier = GridSearchCV(classifier, parameters)
# classifier.fit(features_train, labels_train)
# validated_pred = classifier.predict(features_test)
# validated_acc = accuracy_score(labels_test, validated_pred)
# print 'The Accuracy of GridSearchCV tree is:', round(validated_acc, 3)
# print 'And its parameters are:', classifier.best_params_


