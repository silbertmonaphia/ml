#! /usr/bin/env python
# ecoding:utf-8

from sklearn.base import BaseEstimator
import sklearn.metrics
import numpy
from sklearn.linear_model import LogisticRegression as LR
import time


class SelfLearningModel(BaseEstimator):

    def __init__(self, basemodel, max_iter=200, prob_threshold=0.85):
        self.model = basemodel
        self.max_iter = max_iter
        self.prob_threshold = prob_threshold

    def fit(self, X_labeled, y_labeled, X_unlabeled):

        self.model.fit(X_labeled, y_labeled)
        y_unlabeled = self.model.predict(X_unlabeled)
        unlabeledprob = self.model.predict_proba(X_unlabeled)

        y_unlabeled_old = []

        # re-train, labeling unlabeled instances with model predictions, until
        # convergence
        i = 0

        # if y_unlabeled is equal to  y_unlabeled_old ,it infers that there is
        # no room for optimization,so return
        while (len(y_unlabeled_old) == 0 or numpy.any(y_unlabeled != y_unlabeled_old)) and i < self.max_iter:

            y_unlabeled_old = numpy.copy(y_unlabeled)
            # uidx[tuple] is the number of the row which first or second
            # column's value is better than 0.85
            uidx = numpy.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))
            uidx = uidx[0]  # in order to transform tuple to numpy.ndarray

            # print X_unlabeled[uidx, :].shape
            X_labeled = numpy.vstack((X_labeled, X_unlabeled[uidx, :]))
            y_labeled = numpy.hstack((y_labeled, y_unlabeled_old[uidx]))

            X_unlabeled = numpy.delete(X_unlabeled, uidx, axis=0)
            y_unlabeled = numpy.delete(y_unlabeled_old, uidx, axis=0)
            
            self.model.fit(X_labeled, y_labeled)

            y_unlabeled = self.model.predict(X_unlabeled)
            unlabeledprob = self.model.predict_proba(X_unlabeled)
            i += 1

        return self

    def predict(self, X):
        predict = self.model.predict(X)
        return predict
