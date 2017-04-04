import numpy as np
import random
import copy
from numpy.random import shuffle
import os

class CoTrainingClassifier(object):

    def __init__(self, clf, clf2=None, p=-1, n=-1, k=30, u=75):
        self.clf1_ = clf

        # we will just use a copy of clf (the same kind of classifier) if clf2
        # is not specified
        if clf2 == None:
            self.clf2_ = copy.copy(clf)
        else:
            self.clf2_ = clf2

        # if they only specify one of n or p, through an exception
        if (p == -1 and n != -1) or (p != -1 and n == -1):
            raise ValueError(
                'Current implementation supports either both p and n being specified, or neither')

        self.p_ = p
        self.n_ = n
        self.k_ = k
        self.u_ = u

    def fit(self, X_labeled, y_labeled, X_unlabeled):

        if self.p_ == -1 and self.n_ == -1:
            num_pos = sum(1 for y_i in y_labeled if y_i == 1)
            num_neg = sum(1 for y_i in y_labeled if y_i == 0)

            n_p_ratio = num_neg / float(num_pos)
            # check the proportion of neg and pos in y in order to set n and p
            if n_p_ratio > 1:
                self.p_ = 1
                self.n_ = int(round(self.p_ * n_p_ratio))

            else:
                self.n_ = 1
                self.p_ = int(round(self.n_ / n_p_ratio))
        #n_=1 p_=2
        print 'n_ is',self.n_,'p_ is',self.p_

        assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

        # recreate y_unlabeled
        y_unlabeled = np.full(len(X_unlabeled), -1)

        # [1st time]this is U' in paper
        X_unlabeled_prime = X_unlabeled[-min(len(X_unlabeled), self.u_):, :]
        y_unlabeled_prime = y_unlabeled[-min(len(y_unlabeled), self.u_):]

        # [1st time]remove the samples in U' from U
        X_unlabeled = X_unlabeled[:-len(X_unlabeled_prime), :]
        y_unlabeled = y_unlabeled[:-len(y_unlabeled_prime)]


        iteration = 0  # number of cotraining iterations we've done so far
        while iteration != self.k_ and X_unlabeled.size != 0:

            print 'iteration',iteration
            iteration += 1

            print "Random_Subspace\n"

            length = len(X_labeled)
            X_train = np.vstack((X_labeled,X_unlabeled_prime))
            X_train_view1 = self.random_subspace(X_train)
            X_train_view2 = self.random_subspace(X_train)
            X_labeled_view1 = X_train_view1[:length,:]
            X_unlabeled_view1 = X_train_view1[length:,:]
            X_labeled_view2 = X_train_view2[:length,:]
            X_unlabeled_view2 = X_train_view2[length:,:]

            print "training the clfs with labeled data(view)"
            self.clf1_.fit(X_labeled_view1, y_labeled)
            self.clf2_.fit(X_labeled_view2, y_labeled)

            print "training the clfs with unlabeled data(view)"
            y1 = self.clf1_.predict_proba(X_unlabeled_view1)
            y2 = self.clf2_.predict_proba(X_unlabeled_view2)

            n_index, p_index = [], []

            #to get the n(or p) best negative example's index in y,and put them into the list n(or p)
            n_index_v1 = y1[:,0].argsort()[-self.n_:][::-1]
            p_index_v1 = y1[:,1].argsort()[-self.p_:][::-1]
            n_index_v2 = y2[:,0].argsort()[-self.n_:][::-1]
            p_index_v2 = y2[:,1].argsort()[-self.p_:][::-1]

            n_index.extend(n_index_v1)
            n_index.extend(n_index_v2)
            p_index.extend(p_index_v1)
            p_index.extend(p_index_v2)
            

            print "label the p and n samples of U'\n"
            y_unlabeled_prime[[x for x in p_index]] = 1
            y_unlabeled_prime[[x for x in n_index]] = 0

            print "take those labeled samples out of U' and reform a new set U'' \n"
            id_ = np.where(y_unlabeled_prime != -1)[0]
            X_extent = X_unlabeled_prime[id_, :]
            y_extent = y_unlabeled_prime[id_]

            print "enlarge the previous L with U'' \n"
            X_labeled = np.vstack((X_labeled, X_extent))
            y_labeled = np.hstack((y_labeled, y_extent))

            print "remove p and n samples from U'\n"
            p_old = p_index
            p_index.extend(n_index)
            X_unlabeled_prime = np.delete(X_unlabeled_prime, p_index, axis=0)

            print "add new elements(2p+2n) to U'\n"
            num_to_add = len(p_old) + len(n_index)
            if X_unlabeled.size != 0:
                shuffle(X_unlabeled)
                #shuffle(y_unlabeled)
                X_unlabeled_prime = np.vstack(
                    (X_unlabeled_prime, X_unlabeled[-num_to_add:, :]))
                y_unlabeled_prime = np.hstack(
                    (y_unlabeled_prime, y_unlabeled[-num_to_add:]))
                X_unlabeled = X_unlabeled[:num_to_add, :]
                y_unlabeled = y_unlabeled[:num_to_add]

        print "final training\n"
        self.clf1_.fit(X_labeled, y_labeled)

    def random_subspace(self,X):
        n = X.shape[1]
        m = n // 2
        X_T = X.T
        shuffle(X_T)
        X_T_shuffled_T = X_T.T
        X_view = X_T_shuffled_T[:, :m]
        return X_view

    def predict(self, X):
        y_pred_clf1 = self.clf1_.predict(X)
        return y_pred_clf1