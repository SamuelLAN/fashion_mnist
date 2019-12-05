#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from fashion_mnist_from_git.utils import mnist_reader

random_state = 42

# loading data
X_train, y_train = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='t10k')
classes = np.unique(y_train)


def evaluate(y_true, y_pred, _classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=_classes, average=None)
    f1_micro = f1_score(y_true, y_pred, labels=_classes, average='micro')

    print('\n\n- accuracy: %f\n' % accuracy)
    print('- global f1: %f' % f1_micro)
    for i, _class in enumerate(_classes):
        print('- class %d f1 score: %f' % (_class, f1[i]))


def train_and_predict(_X_train, _y_train, _X_test, _y_test, k=5):
    _knn = KNeighborsClassifier(n_neighbors=k)
    _knn.fit(_X_train, _y_train)

    print('\nStart evaluating training set ...')
    start_time = time.time()
    train_prediction = _knn.predict(_X_train)
    evaluate(train_prediction, _y_train, classes)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating \n')

    print('Start evaluating test set ...')
    start_time = time.time()
    test_prediction = _knn.predict(_X_test)
    evaluate(test_prediction, _y_test, classes)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating \n')


# evaluate original data
train_and_predict(X_train, y_train, X_test, y_test)

for dim_size in range(20, X_train.shape[1], 30):
    print('###################################')
    print('dimension size: %s' % str(dim_size))

    # PCA
    print('Start PCA ...')
    _pca = PCA(n_components=dim_size, random_state=random_state)
    X_train_pca = _pca.fit_transform(X_train)
    X_test_pca = _pca.transform(X_test)
    print('X_train_lda.shape: %s' % repr(X_train_pca.shape))
    print('X_test_lda.shape: %s' % repr(X_test_pca.shape))
    print('Finish PCA')

    for k in range(1, 21, 2):
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('K neighbors: %d' % k)

        # evaluate the data after PCA
        train_and_predict(X_train_pca, y_train, X_test_pca, y_test, k)

# LDA
print('\n------------------------------------')
print('Start LDA ...')
_lda = LDA()
X_train_lda = _lda.fit_transform(X_train, y_train)
X_test_lda = _lda.transform(X_test)
print('X_train_lda.shape: %s' % repr(X_train_lda.shape))
print('X_test_lda.shape: %s' % repr(X_test_lda.shape))
print('Finish LDA')

for k in range(1, 31, 2):
    print('\n@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('K neighbors: %d' % k)

    # evaluate the data after LDA
    train_and_predict(X_train_lda, y_train, X_test_lda, y_test, k)

print('\ndone')
