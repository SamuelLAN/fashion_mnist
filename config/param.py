#!/usr/bin/Python
# -*- coding: utf-8 -*-
import time
from lib.utils import accuracy_one_hot

measure_dict = {
    'accuracy': accuracy_one_hot,
}

NUM_CLASSES = 10

IS_TRAIN = False

RANDOM_STATE = 42

# TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
TIME_DIR = '2019_12_08_11_05_30'
NEW_TIME_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
