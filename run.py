#! /usr/bin/env python
# encoding:utf-8

from scipy.io import arff
from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

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
            if line[-1] == 'pos':
                y.append(1)
            else:
                y.append(0)

            line = list(line)
            line.pop()
            X.append(line)

    # 交叉验证
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    X_unlabeled, X_labeled, y_unlebeled, y_labeled = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    # SL
    print 'start testing SL'

    sl = False
    st = True
    ct = False

    if sl is True:

        clf_SVM = SVC()
        clf_MNB = MultinomialNB()
        clf_DTC = DecisionTreeClassifier()
        clf_SVM.fit(X_labeled, y_labeled)
        clf_MNB.fit(X_labeled, y_labeled)
        clf_DTC.fit(X_labeled, y_labeled)
        print 'Train Over'

        ass_SL_clf(clf_SVM, X_test, y_test)  # 0.7259
        ass_SL_clf(clf_MNB, X_test, y_test)  # 0.7833
        ass_SL_clf(clf_DTC, X_test, y_test)  # 0.7389
        print 'Base Classifier Test Over'

    if st is True:
        pass

        # SSL
        print 'start testing SSL-SelfLearning'

        # Self-Training

        clf_SVM = SVC(probability=True)  #svm has to turn on probability parameter
        ssl_slm_svm = SelfLearningModel(clf_SVM)
        ssl_slm_svm.fit(X_labeled, y_labeled, X_unlabeled)
        print ssl_slm_svm.score(X_test, y_test)
        # 0.5648

        # clf_MNB = MultinomialNB()
        # ssl_slm_mnb = SelfLearningModel(clf_MNB)
        # ssl_slm_mnb.fit(X_labeled, y_labeled, X_unlabeled)
        # print ssl_slm_mnb.score(X_test, y_test)
        # # 0.7981

        # clf_DTC = DecisionTreeClassifier()
        # ssl_slm_dtc = SelfLearningModel(clf_DTC)
        # ssl_slm_dtc.fit(X_labeled, y_labeled, X_unlabeled)
        # print ssl_slm_dtc.score(X_test, y_test)
        # # 0.7259

    if ct is True:
        pass
        # Co-Training

        # ssl_ctc_svm_mnb = CoTrainingClassifier(clf_SVM,clf_MNB)
        # ssl_ctc_svm_mnb.fit(X)
