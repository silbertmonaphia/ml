#! /usr/bin/env python
# ecoding:utf-8

from sklearn.base import BaseEstimator
import sklearn.metrics
import numpy
from sklearn.linear_model import LogisticRegression as LR


class SelfLearningModel(BaseEstimator):
    """
    Self Learning framework for semi-supervised learning

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then iteratively 
    labeles the unlabeled examples with the trained model and then 
    re-trains it using the confidently self-labeled instances 
    (those with above-threshold probability) until convergence.    
    See e.g. http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf
    """

    def __init__(self, basemodel, max_iter=200, prob_threshold=0.85):
        self.model = basemodel
        self.max_iter = max_iter
        self.prob_threshold = prob_threshold

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """Fit base model to the data in a semi-supervised fashion 
        using self training 

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value (-1) for
        unlabeled samples.
        """

        self.model.fit(X_labeled, y_labeled)

        y_unlabeled = self.model.predict(X_unlabeled)
        unlabeledprob = self.predict_proba(X_unlabeled)

        y_unlabeled_old = []

        # re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0

        #if y_unlabeled is equal to  y_unlabeled_old ,it infers that there is no room for optimization,so return
        while (len(y_unlabeled_old) == 0 or numpy.any(y_unlabeled != y_unlabeled_old)) and i < self.max_iter:
            y_unlabeled_old = numpy.copy(y_unlabeled)
            uidx = numpy.where( (unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold) )[0]
            self.model.fit(numpy.vstack( (X_labeled, X_unlabeled[uidx, :]) ), numpy.hstack((y_labeled, y_unlabeled_old[uidx])))
            y_unlabeled = self.model.predict(X_unlabeled)
            unlabeledprob = self.predict_proba(X_unlabeled)

            i += 1

        if not getattr(self.model, "predict_proba", None):
            # Platt scaling if the model cannot generate predictions itself
            self.plattlr = LR()
            preds = self.model.predict(X_labeled)
            self.plattlr.fit(preds.reshape(-1, 1), y_labeled)

        return self

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.
        """

        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            return self.plattlr.predict_proba(preds.reshape(-1, 1))

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.model.predict(X))
