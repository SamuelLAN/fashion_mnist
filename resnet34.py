#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.nn_model_base import NN
from tf_models.resnet34 import ResNet34

keras = tf.keras
RANDOM_STATE = 42


class Model(NN):
    # model param config
    params = {
        **NN.default_params,
        'learning_rate': 5e-6,
        'lr_decay_rate': 0.001,
        'lr_staircase': False,
        'lr_factor': 0.1,
        'lr_patience': 5,
        'batch_size': 64,
        'epoch': 500,
        'early_stop': 30,
        'kernel_initializer': tf.initializers.glorot_uniform(seed=RANDOM_STATE),
        # 'kernel_initializer': 'glorot_uniform',
        'kernel_regularizer': None,
        'dropout': 0.5,
    }



    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            'loss': keras.losses.categorical_crossentropy,
            'metrics': [
                keras.metrics.categorical_accuracy,
                keras.metrics.categorical_crossentropy,
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
            ],
        }

    def build(self):
        """ Build neural network architecture """
        self.model = ResNet34('channels_last', classes=10,
                              kernel_initializer=self.params['kernel_initializer'],
                              kernel_regularizer=self.params['kernel_regularizer'],
                              dropout=self.params['dropout'])
