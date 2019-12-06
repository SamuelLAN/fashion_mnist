#!/usr/bin/Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.nn_model_base import NN

keras = tf.keras
layers = keras.layers
RANDOM_STATE = 42


class Model(NN):
    # model param config
    params = {
        **NN.default_params,
        'learning_rate': 8e-7,
        'lr_decay_rate': 0.001,
        'lr_staircase': False,
        'batch_size': 64,
        'epoch': 300,
        'early_stop': 20,
        'kernel_initializer': tf.initializers.glorot_uniform(seed=RANDOM_STATE),
        'kernel_regularizer': None,
        'dropout': None,
    }

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            'loss': keras.losses.categorical_crossentropy,
            # 'loss': tf.losses.sparse_softmax_cross_entropy,
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
        self.model = keras.Sequential([
            layers.Conv2D(6, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(6, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(6, 3, padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(),
            layers.Conv2D(16, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(16, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(16, 3, padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(64, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(64, 3, padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(128, 3, padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(128, 3, padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.Flatten(),
            layers.Dense(128, activation='relu', kernel_initializer=self.params['kernel_initializer']),
            layers.Dense(10, activation='softmax', kernel_initializer=self.params['kernel_initializer']),
        ])
