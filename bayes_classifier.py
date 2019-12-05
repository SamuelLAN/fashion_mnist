#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from fashion_mnist_from_git.utils import mnist_reader

random_state = 42

# loading data
X_train, y_train = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='t10k')


def mat(*args):
    k = args[0]
    for _arg in args[1:]:
        k = np.matmul(k, _arg)
    return k


class BayesClassifier:
    def __init__(self, _x_train, _y_train):
        self.__x_train = _x_train
        self.__y_train = _y_train
        self.__classes = np.unique(y_train)

        self.__param_list = []
        self.__evaluate_params()

    def __evaluate_params(self):
        epision = 0.001

        for _class in self.__classes:
            data = self.__x_train[self.__y_train == _class]

            prob_w = float(data.shape[0]) / self.__x_train.shape[0]

            mean = np.mean(data, axis=0)
            cov = np.cov(data.transpose())

            # regularization
            cov = cov + epision * np.identity(cov.shape[0], dtype=np.float32)
            det = np.linalg.det(cov)
            inv_cov = np.linalg.inv(cov)

            # pre calculate some variables
            m = mean.reshape((-1, 1))
            mean_inv_cov = np.matmul(m.transpose(), inv_cov)
            mean_inv_cov_mean_05 = np.matmul(mean_inv_cov, m) * 0.5

            self.__param_list.append({
                'class': _class,
                'log_probability_w_i': np.log(prob_w),
                'det': det,
                'inv_cov': inv_cov,
                'mean_inv_cov': mean_inv_cov,
                'mean_inv_cov_mean_05': mean_inv_cov_mean_05,
            })

    def predict(self, x):
        log_posterior_list = []
        for _class in self.__classes:
            params = self.__param_list[_class]
            log_probability_w_i = params['log_probability_w_i']

            x = x.reshape((-1, 1))
            inv_cov = params['inv_cov']
            mean_inv_cov = params['mean_inv_cov']
            mean_inv_cov_mean_05 = params['mean_inv_cov_mean_05']

            # since the det of cov is either 0 or inf, I assume the "log(det)" is the same in each class
            det = 0
            log_likelihood = -0.5 * mat(x.transpose(), inv_cov, x) + \
                             np.matmul(mean_inv_cov, x) - \
                             mean_inv_cov_mean_05

            log_posterior = log_likelihood + log_probability_w_i

            log_posterior_list.append([_class, log_posterior])

        log_posterior_list.sort(key=lambda x: -x[1])
        prediction_class = log_posterior_list[0][0]

        return prediction_class

    def evaluate_dataset(self, X, Y):
        prediction_list = []

        length = len(X)
        for i, x in enumerate(X):
            if i % 100 == 0:
                progress = float(i + 1) / length * 100.0
                print('\rprogress: %.2f%% ' % progress, end='')

            prediction_list.append(self.predict(x))

        y_pred = np.array(prediction_list)

        accuracy = accuracy_score(Y, y_pred)
        f1 = f1_score(Y, y_pred, labels=self.__classes, average=None)
        f1_micro = f1_score(Y, y_pred, labels=self.__classes, average='micro')

        print('\n\n- accuracy: %f\n' % accuracy)
        print('- global f1: %f' % f1_micro)
        for i, _class in enumerate(self.__classes):
            print('- class %d f1 score: %f' % (_class, f1[i]))


def evaluate(_X_train, _y_train, _X_test, _y_test):
    classifier = BayesClassifier(_X_train, _y_train)

    print('\nStart evaluating training set ...')
    start_time = time.time()
    classifier.evaluate_dataset(_X_train, _y_train)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating \n')

    print('Start evaluating test set ...')
    start_time = time.time()
    classifier.evaluate_dataset(_X_test, _y_test)
    end_time = time.time()
    print('use time: %ds \n' % (end_time - start_time))
    print('Finish evaluating \n')


# evaluate original data
evaluate(X_train, y_train, X_test, y_test)

for k in range(20, X_train.shape[1], 30):
    print('###################################')
    print('dimension size: %s' % str(k))

    # PCA
    print('Start PCA ...')
    _pca = PCA(n_components=k, random_state=random_state)
    X_train_pca = _pca.fit_transform(X_train)
    X_test_pca = _pca.transform(X_test)
    print('X_train_lda.shape: %s' % repr(X_train_pca.shape))
    print('X_test_lda.shape: %s' % repr(X_test_pca.shape))
    print('Finish PCA')

    # evaluate the data after PCA
    evaluate(X_train_pca, y_train, X_test_pca, y_test)

# LDA
print('\n------------------------------------')
print('Start LDA ...')
_lda = LDA()
X_train_lda = _lda.fit_transform(X_train, y_train)
X_test_lda = _lda.transform(X_test)
print('X_train_lda.shape: %s' % repr(X_train_lda.shape))
print('X_test_lda.shape: %s' % repr(X_test_lda.shape))
print('Finish LDA')

# evaluate the data after LDA
evaluate(X_train_lda, y_train, X_test_lda, y_test)

print('\ndone')
