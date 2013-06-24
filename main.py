# -*- coding:utf-8 -*-
'''
Created on 24 juin 2013

@author: ediemert
'''

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import metrics
from sklearn.feature_extraction import FeatureHasher


def parse_header(line):
    attributes = line.strip().split(',')
    return attributes

def parse_line(line, attributes):
    features = [ int(_) for _ in line.strip().split(',') ]
    return dict(zip(attributes, features))

def stream_read(fn):
    fd = open(fn)
    attributes = None
    for line in fd:
        if not attributes:
            attributes = parse_header(line)
            continue
        yield parse_line(line, attributes)

def read_train(fn):
    X = []
    y = []
    for d in stream_read(fn):
        y.append(d['ACTION'])
        del d['ACTION']
        X.append(d)
    y = np.array(y)
    #X = FeatureHasher().fit_transform(X)
    X = np.array([np.array([xx for xx in x.itervalues()]) for x in X])
    return X, y

def evaluate(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    fpr, tpr, _thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)

if __name__ == '__main__':

    X, y = read_train('train.csv')
    print("X:", X.shape, "y:",y.shape)
    cross_validation_iterator = cross_validation.StratifiedKFold(y, n_folds=5)
    y_pred = []
    y_true = []
    for train_index, test_index in cross_validation_iterator:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=30, n_jobs=4)
        clf.fit(X_train, y_train)
        y_true.extend(y_test)
        y_pred.extend(clf.predict(X_test))
        print("AUC: %.3f" % evaluate(y_true, y_pred))

