#! /usr/bin/env python
# encoding:utf-8

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
    
from frameworks.SelfLearning import *

#input the data

X = [[0, 0], [1, 1]]

y = [0, 1]


#use SVM as the classifier
clf = SVC(probability=True)

#use Naive Bayes as the classifier
gnb=GaussianNB()

#use Nearest Neighber as the classifier
nbrs=NearestNeighbors()

#SSL
ssmodel = SelfLearningModel(clf)
ssmodel.fit(X, y)