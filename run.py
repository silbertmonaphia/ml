#! /usr/bin/env python
# encoding:utf-8

import numpy as np
import random
from scipy.io import arff
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from selflearning import SelfLearningModel
from cotraining.classifiers import CoTrainingClassifier


def loadData(filepath):
    # feature[[],[]]
    X = []
    # tag['pos','neg']
    y = []

    # load arff file
    with open(filepath, 'rb') as f:
        data, meta = arff.loadarff(f)
        for line in data:
            if line[-1] == 'pos':
                y.append(1)
            else:
                y.append(0)

            line = list(line)
            line.pop()
            X.append(line)

    X = np.array(X)
    y = np.array(y)

    return X, y


def cross_validation(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    X_unlabeled, X_labeled, y_unlebeled, y_labeled = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    return X_labeled, y_labeled, X_unlabeled, X_test, y_test


def ass_baseline(classifier, X_test, y_test):

    point = 0
    prediction = classifier.predict(X_test)
    for i in range(len(prediction)):
        if y_test[i] == prediction[i]:
            point += 1
    accuracy = float(point) / float(len(prediction))
    return accuracy


def test_baseline(X_labeled, y_labeled, X_test, y_test):

    print 'start testing baseline :/'
    clf_SVM = SVC()
    clf_DTC = DecisionTreeClassifier()
    clf_MNB = MultinomialNB()
    
    clf_SVM.fit(X_labeled, y_labeled)
    clf_DTC.fit(X_labeled, y_labeled)
    clf_MNB.fit(X_labeled, y_labeled)
    print 'Train Over'

    print ass_baseline(clf_SVM, X_test, y_test)  # 0.7259
    print ass_baseline(clf_DTC, X_test, y_test)  # 0.7389
    print ass_baseline(clf_MNB, X_test, y_test)  # 0.7833
    print 'Base Classifier Test Over'


def test_selftraing(X_labeled, y_labeled, X_unlabeled, X_test, y_test):

    # SSL-SelfTraining
    print 'start testing SSL-SelfTraining :D'

    # svm has to turn on probability parameter
    clf_SVM = SVC(probability=True)
    ssl_slm_svm = SelfLearningModel(clf_SVM)
    ssl_slm_svm.fit(X_labeled, y_labeled, X_unlabeled)
    print ssl_slm_svm.score(X_test, y_test)
    # 0.5648

    clf_DTC = DecisionTreeClassifier()
    ssl_slm_dtc = SelfLearningModel(clf_DTC)
    ssl_slm_dtc.fit(X_labeled, y_labeled, X_unlabeled)
    print ssl_slm_dtc.score(X_test, y_test)
    # 0.7259

    clf_MNB = MultinomialNB()
    ssl_slm_mnb = SelfLearningModel(clf_MNB)
    ssl_slm_mnb.fit(X_labeled, y_labeled, X_unlabeled)
    print ssl_slm_mnb.score(X_test, y_test)
    # 0.7981


def test_cotraining(X_labeled, y_labeled, X_unlabeled, X_test, y_test):

    # SSL-Co-Training
    print 'start testing SSL-CoTraining :)'

    clf_SVM = SVC(probability=True)
    clf_DTC = DecisionTreeClassifier()
    clf_MNB = MultinomialNB()

    #an object is a class with status,it has memories
    ssl_ctc_svm = CoTrainingClassifier(clf_SVM)
    ssl_ctc_svm.fit(X_labeled,y_labeled,X_unlabeled)
    print ass_baseline(ssl_ctc_svm,X_test,y_test)

    ssl_ctc_dtc = CoTrainingClassifier(clf_DTC)
    ssl_ctc_dtc.fit(X_labeled,y_labeled,X_unlabeled)
    print ass_baseline(ssl_ctc_dtc,X_test,y_test)

    ssl_ctc_mnb = CoTrainingClassifier(clf_MNB)
    ssl_ctc_mnb.fit(X_labeled,y_labeled,X_unlabeled)
    print ass_baseline(ssl_ctc_mnb,X_test,y_test)

    
if __name__ == '__main__':

    # load arff file as X,y ndarray like
    X, y = loadData('./text/JDMilk.arff')

    # Cross validation for 10 times,and compute the average of accuracy
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = cross_validation(X, y)

    test_baseline(X_labeled, y_labeled, X_test, y_test)
    test_selftraing(X_labeled, y_labeled, X_unlabeled, X_test, y_test)
    test_cotraining(X_labeled, y_labeled, X_unlabeled, X_test, y_test)
