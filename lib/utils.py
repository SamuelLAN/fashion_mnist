#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.metrics import roc_auc_score

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]


def k_neighbors(X, k):
    """ Calculate the K nearest neighbors """
    # use cache
    cache_path = os.path.join(PATH_PRJ, 'pre_processing', 'cache', 'k_neighbors_%s.npy' % k)
    if os.path.isfile(cache_path):
        return np.load(cache_path)

    results = []  # save results and return it after the func finishes
    len_x = len(X)

    for i in range(len_x):
        # show the progress of knn
        if i % 10 == 0:
            progress = float(i + 1) / len_x * 100.0
            print('\rk_neighbors progress: %.2f (%d | %d)   ' % (progress, i + 1, len_x), end='')

        cur = X[i]
        _k_neighbors = []  # save the K nearest neighbor

        for j in range(len_x):
            dist = np.sum(np.power(cur - X[j], 2))

            if len(_k_neighbors) <= k:
                _k_neighbors.append([j, dist])
                _k_neighbors.sort(key=lambda x: x[1])
            else:
                if dist < _k_neighbors[-1][1]:
                    _k_neighbors[-1] = [j, dist]
                    _k_neighbors.sort(key=lambda x: x[1])

        # append the nearest neighbor to results
        results.append(_k_neighbors)

    # save results to cache
    np.save(cache_path, results)

    return results


def accuracy(y_true, y_score, pred_prob=None):
    """ calculate the accuracy between outputs and labels; both outputs and labels are 1-D vector """
    return np.sum(y_score == y_true) / float(len(y_true))


def accuracy_one_hot(y_true, y_score, pred_prob=None, axis=-1):
    """ calculate the accuracy between outputs and labels; both outputs and labels are one-hot vector """
    y_true = np.argmax(y_true, axis=axis)
    y_score = np.argmax(y_score, axis=axis)
    return accuracy(y_true, y_score)


def auc(y_true, y_score, pred_prob):
    return roc_auc_score(y_true, pred_prob)


def auc_one_hot(y_true, y_score, pred_prob, axis=-1):
    """ calculate the auc; both outputs and labels are one-hot vector """
    y_true = np.argmax(y_true, axis=axis)
    y_score = np.argmax(y_score, axis=axis)
    return auc(y_true, y_score, pred_prob)


def output_and_log(file_path, output):
    """ Display the output to the console and save it to the log file. """
    # show to the console
    print(output)
    # save to the log file
    with open(file_path, 'ab') as f:
        f.write(output.encode('utf-8'))
