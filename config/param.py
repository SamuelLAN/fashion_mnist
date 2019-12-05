#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
from lib.utils import accuracy_one_hot, auc_one_hot

measure_dict = {
    'accuracy': accuracy_one_hot,
    'auc': auc_one_hot,
}

NUM_CLASSES = 10

IS_TRAIN = True

RANDOM_STATE = 42

TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
# TIME_DIR = '2019_06_03_14_26_43'
