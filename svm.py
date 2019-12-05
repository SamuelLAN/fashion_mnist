#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from fashion_mnist_from_git.utils import mnist_reader

random_state = 42


def evaluate(y_true, y_pred, _classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=_classes, average=None)
    f1_micro = f1_score(y_true, y_pred, labels=_classes, average='micro')

    print('\n\n- accuracy: %f\n' % accuracy)
    print('- global f1: %f' % f1_micro)
    for i, _class in enumerate(_classes):
        print('- class %d f1 score: %f' % (_class, f1[i]))


def train_and_predict(_X_train, _y_train, _X_test, _y_test, c=1., kernel='rbf'):
    o_svc = SVC(c, kernel=kernel, random_state=random_state)

    print('\nStart training model ...')
    start_time = time.time()
    o_svc.fit(_X_train, _y_train)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish training \n')

    print('\nStart evaluating training set ...')
    start_time = time.time()
    train_prediction = o_svc.predict(_X_train)
    evaluate(train_prediction, _y_train, classes)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating training set \n')

    print('Start evaluating test set ...')
    start_time = time.time()
    test_prediction = o_svc.predict(_X_test)
    evaluate(test_prediction, _y_test, classes)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating test set\n')


# loading data
X_train, y_train = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='t10k')
classes = np.unique(y_train)

c_list = [float(i) / 10. for i in range(1, 15, 3)]
kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']

dim_size = 50
c = 1.
kernel = 'linear'

print('###################################')
print('dimension size: %s' % str(dim_size))
print('c: %.2f' % c)
print('kernel: %s' % kernel)

# PCA
print('\nStart PCA ...')
_pca = PCA(n_components=dim_size, random_state=random_state)
X_train_pca = _pca.fit_transform(X_train)
X_test_pca = _pca.transform(X_test)
print('X_train_lda.shape: %s' % repr(X_train_pca.shape))
print('X_test_lda.shape: %s' % repr(X_test_pca.shape))
print('Finish PCA')

train_and_predict(X_train_pca[:10000], y_train[:10000], X_test_pca, y_test, c, kernel)
