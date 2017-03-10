#! /usr/bin/env python
# encoding:utf-8

from scipy.io import arff
from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.semi_supervised

from selflearning import SelfLearningModel
from cotraining.classifiers import CoTrainingClassifier


def ass_SL_clf(classifier, X_test, y_test):

    point = 0
    prediction = classifier.predict(X_test)
    for i in range(len(prediction)):
        if y_test[i] == prediction[i]:
            point += 1
    accuracy = float(point) / float(len(prediction))
    print accuracy
    return accuracy


if __name__ == '__main__':
    
    # feature[[],[]]
    X = []
    # tag['pos','neg']
    y = []

    # load arff file
    with open("./text/JDMilk.arff", 'rb') as f:
        data, meta = arff.loadarff(f)
        for line in data:
            y.append(line[-1])
            line = list(line)
            line.pop()
            X.append(line)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # SL
    clf_SVM = SVC()
    clf_GNB = GaussianNB()
    clf_MNB = MultinomialNB()
    clf_BNB = BernoulliNB()
    clf_DTC = DecisionTreeClassifier()

    test = True

    if test == True:

        clf_SVM.fit(X_train, y_train)
        clf_GNB.fit(X_train, y_train)
        clf_MNB.fit(X_train, y_train)
        clf_BNB.fit(X_train, y_train)
        clf_DTC.fit(X_train, y_train)
        print 'Train Over \n'

        ass_SL_clf(clf_SVM, X_test, y_test)  # 0.8148
        ass_SL_clf(clf_GNB, X_test, y_test)  # 0.7056
        ass_SL_clf(clf_MNB, X_test, y_test)  # 0.85
        ass_SL_clf(clf_BNB, X_test, y_test)  # 0.8926
        ass_SL_clf(clf_DTC, X_test, y_test)  # 0.8296

        print '\nBase Classifier Test Over \n'

    else:
        # SSL
        
        # Self-Training
        ssl_slm_svm = SelfLearningModel(clf_SVM)
        ssl_slm_svm.fit(X_train, y_train)
        print ssl_svm.predict(X_test)

        # Co-Training
        # ssl_ctc_svm_mnb = CoTrainingClassifier(clf_SVM,clf_MNB)
        # ssl_ctc_svm_mnb.fit(X)
