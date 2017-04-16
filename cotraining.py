import numpy as np
import random
import copy
from numpy.random import shuffle
import os


class CoTrainingClassifier(object):

    def __init__(self, clf, p=-1, n=-1, k=200, u=75, threshold=0.92):
        # we will just use a copy of clf (the same kind of classifier) if clf2
        # is not specified
        self.clf1_ = clf
        self.clf2_ = copy.copy(clf)
        self.clf3_ = copy.copy(clf)
        self.clf4_ = copy.copy(clf)
        self.clf5_ = copy.copy(clf)
        self.clf6_ = copy.copy(clf)
        self.clf7_ = copy.copy(clf)
        self.clf8_ = copy.copy(clf)

        self.p_ = p
        self.n_ = n
        self.k_ = k
        self.u_ = u
        self.threshold = threshold

    def fit(self, X_labeled, y_labeled, X_unlabeled):

        self.init_n_p(y_labeled)
        # JDMilk.arff n_=1 p_=2
        #print 'n_ is', self.n_, 'p_ is', self.p_
        assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

        # create y_unlabeled
        y_unlabeled = np.full(len(X_unlabeled), -1)
        # [1st time]this is U' in paper
        X_unlabeled_prime = X_unlabeled[-min(len(X_unlabeled), self.u_):, :]
        y_unlabeled_prime = y_unlabeled[-min(len(y_unlabeled), self.u_):]
        # [1st time]remove the samples in U' from U
        X_unlabeled = X_unlabeled[:-len(X_unlabeled_prime), :]
        y_unlabeled = y_unlabeled[:-len(y_unlabeled_prime)]

        #seed = self.generate_static_seed(X_labeled, X_unlabeled_prime)
        iteration = 0  # number of cotraining iterations we've done so far

        while iteration != self.k_ and X_unlabeled.size != 0:

            print 'iteration', iteration
            iteration += 1

            print "Random_Subspace\n"
            length = len(X_labeled)
            X_train = np.vstack((X_labeled, X_unlabeled_prime))

            X_train_view1 = self.dynamic_random_subspace(X_train)
            X_train_view2 = self.dynamic_random_subspace(X_train)
            X_train_view3 = self.dynamic_random_subspace(X_train)
            X_train_view4 = self.dynamic_random_subspace(X_train)
            # X_train_view5 = self.dynamic_random_subspace(X_train)
            # X_train_view6 = self.dynamic_random_subspace(X_train)
            # X_train_view7 = self.dynamic_random_subspace(X_train)
            # X_train_view8 = self.dynamic_random_subspace(X_train)

            X_labeled_view1 = X_train_view1[:length, :]
            X_unlabeled_view1 = X_train_view1[length:, :]

            X_labeled_view2 = X_train_view2[:length, :]
            X_unlabeled_view2 = X_train_view2[length:, :]

            X_labeled_view3 = X_train_view3[:length, :]
            X_unlabeled_view3 = X_train_view3[length:, :]

            X_labeled_view4 = X_train_view4[:length, :]
            X_unlabeled_view4 = X_train_view4[length:, :]

            # X_labeled_view5 = X_train_view5[:length, :]
            # X_unlabeled_view5 = X_train_view5[length:, :]

            # X_labeled_view6 = X_train_view6[:length, :]
            # X_unlabeled_view6 = X_train_view6[length:, :]

            # X_labeled_view7 = X_train_view7[:length, :]
            # X_unlabeled_view7 = X_train_view7[length:, :]

            # X_labeled_view8 = X_train_view8[:length, :]
            # X_unlabeled_view8 = X_train_view8[length:, :]

            print "training the clfs with labeled data(view)"
            self.clf1_.fit(X_labeled_view1, y_labeled)
            self.clf2_.fit(X_labeled_view2, y_labeled)
            self.clf3_.fit(X_labeled_view3, y_labeled)
            self.clf4_.fit(X_labeled_view4, y_labeled)
            # self.clf5_.fit(X_labeled_view5, y_labeled)
            # self.clf6_.fit(X_labeled_view6, y_labeled)
            # self.clf7_.fit(X_labeled_view7, y_labeled)
            # self.clf8_.fit(X_labeled_view8, y_labeled)

            # how to judge whether these 4 classifiers change ?
            print "predict probability"
            p1 = self.clf1_.predict_proba(X_unlabeled_view1)
            p2 = self.clf2_.predict_proba(X_unlabeled_view2)
            p3 = self.clf3_.predict_proba(X_unlabeled_view3)
            p4 = self.clf4_.predict_proba(X_unlabeled_view4)
            # p5 = self.clf5_.predict_proba(X_unlabeled_view5)
            # p6 = self.clf6_.predict_proba(X_unlabeled_view6)
            # p7 = self.clf7_.predict_proba(X_unlabeled_view7)
            # p8 = self.clf8_.predict_proba(X_unlabeled_view8)

            n_index, p_index = [], []
            # to get the n(or p) best negative example's index in y,and put
            # them into the list n(or p)

            # n_index_v1 = list(p1[:, 0].argsort()[-self.n_:][::-1])
            # p_index_v1 = list(p1[:, 1].argsort()[-self.p_:][::-1])
            # n_index_v2 = list(p2[:, 0].argsort()[-self.n_:][::-1])
            # p_index_v2 = list(p2[:, 1].argsort()[-self.p_:][::-1])
            # n_index_v3 = list(p3[:, 0].argsort()[-self.n_:][::-1])
            # p_index_v3 = list(p3[:, 1].argsort()[-self.p_:][::-1])
            # n_index_v4 = list(p4[:, 0].argsort()[-self.n_:][::-1])
            # p_index_v4 = list(p4[:, 1].argsort()[-self.p_:][::-1])

            n_index_v1 = np.where(p1[:, 0] > self.threshold)[0]
            p_index_v1 = np.where(p1[:, 1] > self.threshold)[0]

            n_index_v2 = np.where(p2[:, 0] > self.threshold)[0]
            p_index_v2 = np.where(p2[:, 1] > self.threshold)[0]

            n_index_v3 = np.where(p3[:, 0] > self.threshold)[0]
            p_index_v3 = np.where(p3[:, 1] > self.threshold)[0]

            n_index_v4 = np.where(p4[:, 0] > self.threshold)[0]
            p_index_v4 = np.where(p4[:, 1] > self.threshold)[0]

            # n_index_v5 = np.where(p5[:, 0] > self.threshold)[0]
            # p_index_v5 = np.where(p5[:, 1] > self.threshold)[0]

            # n_index_v6 = np.where(p6[:, 0] > self.threshold)[0]
            # p_index_v6 = np.where(p6[:, 1] > self.threshold)[0]

            # n_index_v7 = np.where(p7[:, 0] > self.threshold)[0]
            # p_index_v7 = np.where(p7[:, 1] > self.threshold)[0]

            # n_index_v8 = np.where(p8[:, 0] > self.threshold)[0]
            # p_index_v8 = np.where(p8[:, 1] > self.threshold)[0]

            n_index.extend(n_index_v1)
            # duck type (although n_index_v1 is ndarray, and n_index is list, but
            # ndarray is iterative, and list.extend() just need an iterative
            # object as its input)
            p_index.extend(p_index_v1)

            n_index.extend(n_index_v2)
            p_index.extend(p_index_v2)

            n_index.extend(n_index_v3)
            p_index.extend(p_index_v3)

            n_index.extend(n_index_v4)
            p_index.extend(p_index_v4)

            # n_index.extend(n_index_v5)
            # p_index.extend(p_index_v5)

            # n_index.extend(n_index_v6)
            # p_index.extend(p_index_v6)

            # n_index.extend(n_index_v7)
            # p_index.extend(p_index_v7)

            # n_index.extend(n_index_v8)
            # p_index.extend(p_index_v8)

            # remove those repeat(get the union of 4 veiws)
            # but actually the repeat does not matter a lot in accuracy
            n_index = list(set(n_index))
            p_index = list(set(p_index))

            if len(n_index) == 0 and len(p_index) == 0:
                continue
            print "label(fake) the p and n samples\n"
            y_unlabeled_prime[p_index] = 1
            y_unlabeled_prime[n_index] = 0

            print "take those labeled samples out of U' and reconstruct a new set U'' \n"
            id_ = np.where(y_unlabeled_prime != -1)[0]
            X_extent = X_unlabeled_prime[id_, :]
            y_extent = y_unlabeled_prime[id_]

            print "enlarge the previous training data set L with set U'' \n"
            X_labeled = np.vstack((X_labeled, X_extent))
            y_labeled = np.hstack((y_labeled, y_extent))

            print "remove p and n samples from U'\n"
            X_unlabeled_prime = np.delete(X_unlabeled_prime, id_, axis=0)
            y_unlabeled_prime = np.delete(y_unlabeled_prime, id_, axis=0)

            print "replenish new samples to U' from unlabeled data set U \n"
            num_to_add = len(id_)
            if X_unlabeled.size != 0:
                shuffle(X_unlabeled)
                # shuffle(y_unlabeled)
                X_unlabeled_prime = np.vstack(
                    (X_unlabeled_prime, X_unlabeled[-num_to_add:, :]))
                y_unlabeled_prime = np.hstack(
                    (y_unlabeled_prime, y_unlabeled[-num_to_add:]))
                X_unlabeled = X_unlabeled[:-num_to_add, :]
                y_unlabeled = y_unlabeled[:-num_to_add]
            if X_unlabeled.size == 0:
                print 'X_unlabeled is Empty, OVER'

        print 'the final size of X_labeled is ', X_labeled.shape
        print "final training\n"
        self.clf1_.fit(X_labeled, y_labeled)

    def init_n_p(self, y_labeled):
        """
        get n and p value according to the negtive examples and positive examples 
        ratio of train data set
        """

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
        return self

    def dynamic_random_subspace(self, X):

        print 'dynamic'
        # n means the total number of the input X's features
        n = X.shape[1]
        # m means every subspace shoud include m features
        m = n / 2
        print 'm is ', m
        X_T = X.T
        shuffle(X_T)
        X_T_shuffled_T = X_T.T
        X_view = X_T_shuffled_T[:, :m]
        return X_view

    def generate_static_seed(self, X_labeled, X_unlabeled_prime):

        X_train = np.vstack((X_labeled, X_unlabeled_prime))
        X_T = X_train.T
        seed = np.arange(X_T.shape[0])
        shuffle(seed)
        return seed

    def static_random_subspace(self, X, seed):

        print 'static'
        # n means the total number of the input X's features
        n = X.shape[1]
        # m means every subspace shoud include m features
        m = n / 2
        X_T = X.T
        X_T_shuffled = X_T[seed]
        X_T_shuffled_T = X_T_shuffled.T
        X_view = X_T_shuffled_T[:, :m]
        return X_view

    def predict(self, X):
        y_pred_clf1 = self.clf1_.predict(X)
        return y_pred_clf1
