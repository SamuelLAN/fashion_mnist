#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.nn_model_base import NN
from tf_models.resnet50 import ResNet50

keras = tf.keras
RANDOM_STATE = 42


class Model(NN):
    # model param config
    params = {
        **NN.default_params,
        'learning_rate': 1e-2,
        'lr_decay_rate': 0.7,
        'lr_staircase': True,
        'batch_size': 20,
        'epoch': 200,
        'early_stop': 20,
        'kernel_initializer': tf.initializers.glorot_uniform(seed=RANDOM_STATE),
        'kernel_regularizer': None,
        'dropout': 0.5,
    }

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            'loss': keras.losses.binary_crossentropy,
            'metrics': [
                keras.metrics.binary_accuracy,
                keras.metrics.binary_crossentropy,
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
            ],
        }

    def build(self):
        """ Build neural network architecture """
        self.model = ResNet50('channels_last', classes=10,
                              kernel_initializer=self.params['kernel_initializer'],
                              kernel_regularizer=self.params['kernel_regularizer'],
                              dropout=self.params['dropout'])
