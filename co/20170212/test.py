#! /usr/bin/env python
# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import logging
import random
import multiprocessing

import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import jieba


def load_data(filename):
    data = open(filename, 'r').readlines()
    #datat is a list
    for data in data:
        print data
    return data, len(data)


def split_data(data, data_label, size=0.2, random=0):
    train, test, train_label, test_label = train_test_split(data, data_label, test_size=size, random_state=random)
    return train, test, train_label, test_label


def naive_bayes_classifier(train_x, train_y):    
    model = MultinomialNB()
    model.fit(train_x, train_y)
    return model


def Tfidf(data):
    part_word = lambda x: jieba.cut(x)
    stopwords = open('stopword.txt', 'r').readlines()
    vec = TfidfVectorizer(tokenizer=part_word, stop_words=stopwords)
    vector=vec.fit_transform(data).toarray()
    return vector


def delete(datas, vectors):
    index = []
    for i in range(datas.shape[0]):
        for vector in vectors:
            if not (vector - datas[i, :]).any():
                index.append(i)
    return np.delete(datas, index, 0)


def trainer(param_wrapper):

    #param_wrapper = unsplit, train, train_label, m
    unsplit = param_wrapper[0]
    train = param_wrapper[1]
    train_label = param_wrapper[2]
    m = param_wrapper[3]

    return
    pro_0 = np.array([])
    pro_1 = np.array([])

    for m_i in range(m):

        #var 'rand' stands for the index of those features which is going to construct subspace
        rand = random.sample(range(train.shape[1]), train.shape[1] / m)
        model = naive_bayes_classifier(train[:, np.array(rand)], train_label)
        #unsplit[rand] means a view
        pro = model.predict_proba(unsplit[rand])

        pro_0 = np.r_[pro_0, pro[:, 0]]#negative
        pro_1 = np.r_[pro_1, pro[:, 1]]#positive

    if max(pro_0) > max(pro_1):

        return max(pro_0), 0, unsplit

    else:

        return max(pro_1), 1, unsplit


def main(n,m):

    # global n, m

    # load data
    pos, pos_size = load_data('pos.txt')
    neg, neg_size = load_data('neg.txt')
    data = pos + neg

    #generate labels of pos and neg
    label = [1] * pos_size + [0] * neg_size

    #vectorize list==>ndarray
    vector = Tfidf(data)

    # split data,train=0.1,test=0.2,unsplit=0.7
    train, test, train_label, test_label = split_data(vector, label)
    unsplits, train, unsplit_label, train_label = split_data(train, train_label, 1.0 / 8)

    while unsplits.shape[0] != 0:
        param_wrapper = []
        for unsplit in unsplits:
            param_wrapper.append((unsplit, train, train_label, m))

        #doesn't change the train and train_label set(not enlarge them)
        results = pool.map(trainer, param_wrapper)
        

        pro_0 = []
        pro_1 = []
        vectors_0 = []
        vectors_1 = []

        #divide result into pos and neg these 2 classes
        for result in results:

            if result[1] > 0:#if the tag is pos
                pro_1.append(result[0])
                vectors_1.append(result[2])
            else:
                pro_0.append(result[0])
                vectors_0.append(result[2])

        index_0 = np.argsort(-np.array(pro_0))
        index_1 = np.argsort(-np.array(pro_1))
        vectors_0 = np.array(vectors_0)
        vectors_1 = np.array(vectors_1)

        #Update train and train_label set and remove 
        if vectors_0.shape[0] >= n:
            train = np.r_[train, vectors_0[index_0[0:n], :]]
            train_label = np.r_[train_label, np.array([0] * n)]
            unsplits = delete(unsplits, vectors_0[index_0[0:n], :])

        else:
            if vectors_0.shape[0] > 0:
                train = np.r_[train, vectors_0]
                train_label = np.r_[train_label,
                                    np.array([0] * vectors_0.shape[0])]
                unsplits = delete(unsplits, vectors_0)

        if vectors_1.shape[0] >= n:
            train = np.r_[train, vectors_1[index_1[0:n], :]]
            train_label = np.r_[train_label, np.array([1] * n)]
            unsplits = delete(unsplits, vectors_1[index_1[0:n], :])

        else:

            if vectors_1.shape[0] > 0:
                train = np.r_[train, vectors_1]
                train_label = np.r_[train_label,
                                    np.array([1] * vectors_1.shape[0])]
                unsplits = delete(unsplits, vectors_1)

        print 'unsplit= ', str(unsplits.shape[0]), 'train= ', str(train.shape[0])

    model = naive_bayes_classifier(train, train_label)

    #test
    predict = model.predict(test)
    accuracy = metrics.accuracy_score(test_label, predict)
    recall = metrics.recall_score(test_label, predict)
    print 'accuracy= ' + str(accuracy * 100) + '%'
    print 'recall= ' + str(recall * 100) + '%'


if __name__ == '__main__':
    # set param
    m = 4  # the number of subspaces
    n = 8  # the number of updates staff every time
    pool = multiprocessing.Pool(processes=14)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='my.log',
                        filemode='w')
    main(n,m)
