#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from fashion_mnist_from_git.utils import mnist_reader
from config.path import PATH_SVM_LOG
from lib.utils import output_and_log

random_state = 42


def echo(string):
    output_and_log(PATH_SVM_LOG, string)


def load_data():
    """ split data into training set, validation set and test set """
    X_train_all, y_train_all = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='t10k')
    _classes = np.unique(y_train_all)

    # convert to np.array
    X_train_all = np.array(X_train_all)
    y_train_all = np.array(y_train_all)

    # ready for split data
    data_length = len(X_train_all)
    random_indices = np.array(range(data_length))

    # set random seed
    random.seed(random_state)
    random.shuffle(random_indices)

    train_end_index = int(0.9 * data_length)
    X_train = X_train_all[random_indices[: train_end_index]]
    y_train = y_train_all[random_indices[: train_end_index]]
    X_val = X_train_all[random_indices[train_end_index:]]
    y_val = y_train_all[random_indices[train_end_index:]]
    del X_train_all
    del y_train_all

    return X_train, y_train, X_val, y_val, X_test, y_test, _classes


def evaluate(y_true, y_pred, _classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=_classes, average=None)
    f1_micro = f1_score(y_true, y_pred, labels=_classes, average='micro')

    echo('\n- accuracy: %f\n' % accuracy)
    echo('- global f1: %f' % f1_micro)
    for i, _class in enumerate(_classes):
        echo('- class %d f1 score: %f' % (_class, f1[i]))


def train_and_predict(_X_train, _y_train, _X_val, _y_val, _X_test, _y_test, c=1., kernel='rbf', **kwargs):
    o_svc = SVC(c, kernel=kernel, random_state=random_state, tol=kwargs['tol'], gamma=kwargs['gamma'])

    echo('\nStart training model ...')
    start_time = time.time()

    o_svc.fit(_X_train, _y_train)

    end_time = time.time()
    echo('use time: %ds \n' % (end_time - start_time))
    echo('Finish training \n')

    echo('\nStart evaluating training set ...')
    start_time = time.time()

    train_prediction = o_svc.predict(_X_train)
    evaluate(train_prediction, _y_train, classes)

    end_time = time.time()
    echo('use time: %ds \n' % (end_time - start_time))
    echo('Finish evaluating training set \n')

    echo('Start evaluating validation set ...')
    start_time = time.time()

    val_prediction = o_svc.predict(_X_val)
    evaluate(val_prediction, _y_val, classes)

    end_time = time.time()
    echo('use time: %ds \n' % (end_time - start_time))
    echo('Finish evaluating validation set\n')

    echo('Start evaluating test set ...')
    start_time = time.time()

    test_prediction = o_svc.predict(_X_test)
    evaluate(test_prediction, _y_test, classes)

    end_time = time.time()
    echo('use time: %ds \n' % (end_time - start_time))
    echo('Finish evaluating test set\n')


def svm(_dim_size, _c, _kernel, _train_sample_num, _reduction_method='pca', **kwargs):
    # show and log the parameters
    echo('-----------------------------------------------------')
    echo('start_time: %s' % str(time.strftime('%Y.%m.%d %H:%M:%S')))
    echo('dimension size: %s' % str(_dim_size))
    echo('c: %.2f' % _c)
    echo('kernel: %s' % _kernel)
    echo('train_sample_num: %d' % _train_sample_num)
    echo('reduction_method: %s' % _reduction_method)

    if _reduction_method == 'pca':
        # PCA
        echo('\nStart PCA ...')

        _pca = PCA(n_components=_dim_size, random_state=random_state)
        X_train_reduced = _pca.fit_transform(X_train)
        X_val_reduced = _pca.transform(X_val)
        X_test_reduced = _pca.transform(X_test)

        echo('X_train_pca.shape: %s' % repr(X_train_reduced.shape))
        echo('X_val_pca.shape: %s' % repr(X_val_reduced.shape))
        echo('X_test_pca.shape: %s' % repr(X_test_reduced.shape))
        echo('Finish PCA')

    else:

        # LDA
        echo('\nStart LDA ...')
        _lda = LDA()
        X_train_reduced = _lda.fit_transform(X_train, y_train)
        X_val_reduced = _lda.transform(X_val)
        X_test_reduced = _lda.transform(X_test)
        echo('X_train_lda.shape: %s' % repr(X_train_reduced.shape))
        echo('X_val_lda.shape: %s' % repr(X_val_reduced.shape))
        echo('X_test_lda.shape: %s' % repr(X_test_reduced.shape))
        echo('Finish LDA')

        _train_sample_num = len(X_train_reduced)

    echo('\nUse %d samples for training ' % _train_sample_num)

    train_and_predict(X_train_reduced[:_train_sample_num], y_train[:_train_sample_num], X_val_reduced, y_val,
                      X_test_reduced, y_test,
                      _c, _kernel, **kwargs)


# loading data
X_train, y_train, X_val, y_val, X_test, y_test, classes = load_data()

# define the parameters that need to test
c_list = [float(i) / 10. for i in range(10, 41, 3)]
# kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_list = ['rbf']
gamma_list = ['auto', 'scale', 0.3, 0.1, 0.01, 0.03, 0.06, 0.001, 0.003, 0.006]
tol_list = [1e-3, 1e-4, 1e-5]

# c = 1.
train_sample_num = 54000
reduction_method = 'lda'
dim_size = 9 if reduction_method == 'lda' else 50

for kernel in kernel_list:
    for c in c_list:
        for gamma in gamma_list:
            for tol in tol_list:
                params = {'tol': tol, 'gamma': gamma}
                svm(dim_size, c, kernel, train_sample_num, reduction_method, **params)

print('\ndone')
