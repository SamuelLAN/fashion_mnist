#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
import random
import numpy as np
from resnet50 import Model
from lib.utils import output_and_log
from config import path
from config.param import TIME_DIR, RANDOM_STATE
from fashion_mnist_from_git.utils import mnist_reader


class Train:
    """ Run the model and estimate its performance """

    MODEL_CLASS = Model
    MODEL_DIR = TIME_DIR

    def __init__(self):
        # initialize data instances
        self.__X_train_all, self.__y_train_all = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion',
                                                                         kind='train')
        self.__X_test, self.__y_test = mnist_reader.load_mnist('fashion_mnist_from_git/data/fashion', kind='t10k')

        self.__normalize()
        self.__reshape_data()
        self.__split_data()

    def __normalize(self):
        self.__X_train_all = self.__X_train_all / 255.
        self.__X_test = self.__X_train_all / 255.

    def __reshape_data(self):
        self.__X_train_all = self.__X_train_all.reshape((-1, 28, 28, 1))
        self.__X_test = self.__X_test.reshape((-1, 28, 28, 1))

        self.__y_train_all = np.eye(10)[self.__y_train_all]
        self.__y_test = np.eye(10)[self.__y_test]

    def __split_data(self):
        """ split data into training set, validation set and test set """
        # ready for split data
        data_length = len(self.__X_train_all)
        random_indices = np.array(range(data_length))

        # set random seed
        random.seed(RANDOM_STATE)
        random.shuffle(random_indices)

        train_end_index = int(0.9 * data_length)
        self.__X_train = self.__X_train_all[random_indices[: train_end_index]]
        self.__y_train = self.__y_train_all[random_indices[: train_end_index]]
        self.__X_val = self.__X_train_all[random_indices[train_end_index:]]
        self.__y_val = self.__y_train_all[random_indices[train_end_index:]]
        del self.__X_train_all
        del self.__y_train_all

    def run(self):
        """ train model """
        print('\nStart training model %s/%s ...' % (self.MODEL_DIR, path.TRAIN_MODEL_NAME))

        # initialize model instance
        model = self.MODEL_CLASS(self.MODEL_DIR, path.TRAIN_MODEL_NAME)

        # train model
        train_start_time = time.time()
        val_result_dict = model.train(self.__X_train, self.__y_train, self.__X_val, self.__y_val)
        train_use_time = time.time() - train_start_time

        # test model
        eval_train_start_time = time.time()
        train_result_dict = model.test(None, None, self.__X_train, self.__y_train, 'train')
        eval_train_end_time = time.time()
        test_result_dict = model.test(self.__X_train, self.__y_train, self.__X_test, self.__y_test, 'test')
        eval_test_use_time = time.time() - eval_train_end_time
        eval_train_time = eval_train_end_time - eval_train_start_time

        print('\nFinish training\n')

        # show results
        self.__log_results(self.MODEL_DIR, train_result_dict, val_result_dict, test_result_dict,
                           self.MODEL_CLASS.params, train_use_time, eval_train_time, eval_test_use_time)

    @staticmethod
    def __log_results(model_time, train_result_dict, val_result_dict, test_result_dict, model_params,
                      train_use_time, eval_train_time, eval_test_use_time):
        """
        Show the validation result
         as well as the model params to console and save them to the log file.
        """

        data = (path.MODEL_NAME,
                model_time,
                train_result_dict,
                val_result_dict,
                test_result_dict,
                model_params,
                str(train_use_time),
                str(eval_train_time),
                str(eval_test_use_time),
                time.strftime('%Y.%m.%d %H:%M:%S'))

        output = 'model_name: %s\n' \
                 'model_time: %s\n' \
                 'train_result_dict: %s\n' \
                 'val_result_dict: %s\n' \
                 'test_result_dict: %s\n' \
                 'model_params: %s\n' \
                 'train_use_time: %ss\n' \
                 'eval_train_time: %ss\n' \
                 'eval_test_use_time: %ss\n' \
                 'time: %s\n\n' % data

        # show and then save result to log file
        output_and_log(path.PATH_MODEL_LOG, output)


o_train = Train()
o_train.run()
