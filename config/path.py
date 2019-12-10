#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os

MODEL_NAME = 'VGG16_with_bn'
TRAIN_MODEL_NAME = MODEL_NAME

__CUR_DIR = os.path.abspath(os.path.split(__file__)[0])
ROOT_DIR = os.path.split(__CUR_DIR)[0]

# directory for saving the runtime files
__RUNTIME_DIR = os.path.join(ROOT_DIR, 'runtime')

# directory for saving models
__PATH_MODEL_DIR = os.path.join(__RUNTIME_DIR, 'models')
PATH_MODEL_DIR = os.path.join(__PATH_MODEL_DIR, MODEL_NAME)

# directory for saving the tensorboard log files
__PATH_BOARD_DIR = os.path.join(__RUNTIME_DIR, 'tensorboard')
PATH_BOARD_DIR = os.path.join(__PATH_BOARD_DIR, MODEL_NAME)

# the log file path, record all the models results and params
PATH_MODEL_LOG = os.path.join(__RUNTIME_DIR, 'model.log')
PATH_SVM_LOG = os.path.join(__RUNTIME_DIR, 'svm.log')


def mkdir_time(upper_path, _time):
    """ create directory with time (for save model) """
    dir_path = os.path.join(upper_path, _time)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def mkdir_if_not_exist(dir_list):
    """ create the directories listed above """
    for dir_path in dir_list:
        if os.path.isfile(dir_path) or os.path.isdir(dir_path):
            continue
        os.mkdir(dir_path)


# create the directories listed above
mkdir_if_not_exist([
    __RUNTIME_DIR,
    __PATH_MODEL_DIR,
    PATH_MODEL_DIR,
    __PATH_BOARD_DIR,
    PATH_BOARD_DIR,
])
