#! /usr/bin/env python
# encoding:utf-8

import numpy as np
import random

from scipy.io import arff
from sklearn.cross_validation import StratifiedKFold  # balanced better!
from sklearn.cross_validation import train_test_split  # not balanced
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from selflearning import SelfLearningModel
from cotraining import CoTrainingClassifier


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

    skf1 = StratifiedKFold(y, n_folds=4)
    for train_index, test_index in skf1:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        skf2 = StratifiedKFold(y_train, n_folds=75)
        for unlabeled_index, labeled_index in skf2:
            X_unlabeled, X_labeled = X[unlabeled_index], X[labeled_index]
            y_unlabeled, y_labeled = y[unlabeled_index], y[labeled_index]
            break

        yield X_labeled, y_labeled, X_unlabeled, X_test, y_test


def evaluation(y_test, predict, accuracyonly=True):

    accuracy = metrics.accuracy_score(y_test, predict)
    if not accuracyonly:
        # can print out precision recall and f1
        print metrics.classification_report(y_test, predict)
    return accuracy


def plot():
    plt.plot([1, 2, 3, 4])
    plt.ylabel('some numbers')
    plt.show()


def test_baseline(X_labeled, y_labeled, X_test, y_test):

    clf_MNB = MultinomialNB()
    clf_SVM = SVC(probability=True)
    clf_DTC = DecisionTreeClassifier()
    print '\nstart testing baseline :/'

    print 'naive bayes'
    clf_MNB.fit(X_labeled, y_labeled)
    predict = clf_MNB.predict(X_test)
    accuracy_bl_mnb = evaluation(y_test, predict)

    print 'svm'
    clf_SVM.fit(X_labeled, y_labeled)
    predict = clf_SVM.predict(X_test)
    accuracy_bl_svm = evaluation(y_test, predict)

    print 'decision tree'
    clf_DTC.fit(X_labeled, y_labeled)
    predict = clf_DTC.predict(X_test)
    accuracy_bl_dtc = evaluation(y_test, predict)

    return accuracy_bl_mnb, accuracy_bl_svm, accuracy_bl_dtc


def test_selftraing(X_labeled, y_labeled, X_unlabeled, X_test, y_test):

    # SSL-SelfTraining
    clf_MNB = MultinomialNB()
    clf_SVM = SVC(probability=True)
    clf_DTC = DecisionTreeClassifier()

    print '\nstart testing SSL-SelfTraining :D'

    print 'naive bayes'
    ssl_slm_mnb = SelfLearningModel(clf_MNB)
    ssl_slm_mnb.fit(X_labeled, y_labeled, X_unlabeled)
    predict = ssl_slm_mnb.predict(X_test)
    accuracy_sf_mnb = evaluation(y_test, predict)

    # svm has to turn on probability parameter
    print 'svm'
    ssl_slm_svm = SelfLearningModel(clf_SVM)
    ssl_slm_svm.fit(X_labeled, y_labeled, X_unlabeled)
    predict = ssl_slm_svm.predict(X_test)
    accuracy_sf_svm = evaluation(y_test, predict)

    print 'decision tree'
    ssl_slm_dtc = SelfLearningModel(clf_DTC)
    ssl_slm_dtc.fit(X_labeled, y_labeled, X_unlabeled)
    predict = ssl_slm_dtc.predict(X_test)
    accuracy_sf_dtc = evaluation(y_test, predict)

    return accuracy_sf_mnb, accuracy_sf_svm, accuracy_sf_dtc


def test_cotraining(X_labeled, y_labeled, X_unlabeled, X_test, y_test):

    # SSL-Co-Training
    print '\nstart testing SSL-CoTraining :)'

    clf_SVM = SVC(probability=True)
    clf_DTC = DecisionTreeClassifier()
    clf_MNB = MultinomialNB()

    print 'naive bayes'
    ssl_ctc_mnb = CoTrainingClassifier(clf_MNB)
    ssl_ctc_mnb.fit(X_labeled, y_labeled, X_unlabeled)
    predict_clf1, predict_clf2 = ssl_ctc_mnb.predict(X_test)
    accuracy_co_mnb = evaluation(y_test, predict_clf1)

    # an object is a class with status,it has memories
    print 'svm'
    ssl_ctc_svm = CoTrainingClassifier(clf_SVM)
    ssl_ctc_svm.fit(X_labeled, y_labeled, X_unlabeled)
    predict_clf1, predict_clf2 = ssl_ctc_svm.predict(X_test)
    accuracy_co_svm = evaluation(y_test, predict_clf1)

    print 'decision tree'
    ssl_ctc_dtc = CoTrainingClassifier(clf_DTC)
    ssl_ctc_dtc.fit(X_labeled, y_labeled, X_unlabeled)
    predict_clf1, predict_clf2 = ssl_ctc_dtc.predict(X_test)
    accuracy_co_dtc = evaluation(y_test, predict_clf1)

    return accuracy_co_mnb, accuracy_co_svm, accuracy_co_dtc


if __name__ == '__main__':

    # the number of experitments
    experitments = 3

    # load arff file as X,y ndarray like
    X, y = loadData('./text/JDMilk.arff')

    # labeled 1%,unlabeled 74%,test 25%
    cv_generator = cross_validation(X, y)

    accuracy_bl = np.zeros((0, 3))
    accuracy_sf = np.zeros((0, 3))
    accuracy_co = np.zeros((0, 3))

    # Cross validation for 10 times,and compute the average of accuracy
    for i in range(experitments):
        print '=' * 10, str(i), 'time'
        X_labeled, y_labeled, X_unlabeled, X_test, y_test = cv_generator.next()
        accuracy_bl = np.vstack((accuracy_bl, np.asarray(test_baseline(X_labeled, y_labeled, X_test, y_test))))
        #accuracy_sf = np.vstack((accuracy_sf, np.asarray(test_selftraing(X_labeled, y_labeled, X_unlabeled, X_test, y_test))))
        accuracy_co = np.vstack((accuracy_co, np.asarray(test_cotraining(X_labeled, y_labeled, X_unlabeled, X_test, y_test))))


    print '\n.... final static average ....\n'
    clfs = ['mnb', 'svm', 'dtc']
    for i,clf in enumerate(clfs):
        print clf
        print 'baseline: ',sum(accuracy_bl[:,i])/float(len(accuracy_bl[:,i]))
        #print 'selftraining: ',sum(accuracy_sf[:,i])/float(len(accuracy_sf[:,i]))
        print 'cotraining: ',sum(accuracy_co[:,i])/float(len(accuracy_co[:,i]))
   