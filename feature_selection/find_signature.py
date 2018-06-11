#!/C:/Python27

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.

words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r") )
authors = pickle.load( open(authors_file, "r") )


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

from sklearn import model_selection
features_train, features_test, labels_train, labels_test = \
    model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to over-fit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, predictions)
print 'The Accuracy is:', round(acc, 3)

f_imp = list(clf.feature_importances_)
imp_count = 0
for importance in f_imp:
    if importance >= 0.2:
        imp_count += 1
print 'Number of important words is:', imp_count
print 'The highest importance is:', round(max(f_imp), 3)
max_index = f_imp.index(max(f_imp))
print 'And its number is:', max_index

f_names = vectorizer.get_feature_names()
print 'The maximum importance goes with this word:', f_names[max_index]
