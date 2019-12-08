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
        'learning_rate': 8e-6,
        'lr_decay_rate': 0.001,
        'lr_staircase': False,
        'lr_factor': 0.1,
        'lr_patience': 5,
        'batch_size': 64,
        'epoch': 500,
        'early_stop': 30,
        'initial_epoch': 0,
        'monitor_start_train_acc': 0.85,
        'kernel_initializer': tf.initializers.glorot_uniform(seed=RANDOM_STATE),
        'kernel_regularizer': None,
        'dropout': 0.5,
    }

    @property
    def config_for_keras(self):
        return {
            'optimizer': tf.train.AdamOptimizer,
            # 'optimizer': keras.optimizers.Adam,
            'loss': keras.losses.categorical_crossentropy,
            # 'loss': tf.losses.sparse_softmax_cross_entropy,
            'metrics': [
                keras.metrics.categorical_accuracy,
                keras.metrics.categorical_crossentropy,
            ],
            'callbacks': [
                self.callback_tf_board,
                self.callback_saver,
                # self.callback_reduce_lr,
            ],
        }

    def build(self):
        """ Build neural network architecture """
        self.model = keras.Sequential([
            layers.Conv2D(6, 3, name='conv1_1', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(6, 3, name='conv1_2', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(6, 3, name='conv1_3', padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(name='bn1'),
            layers.Conv2D(16, 3, name='conv2_1', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(16, 3, name='conv2_2', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(16, 3, name='conv2_3', padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(name='bn2'),
            layers.Conv2D(64, 3, name='conv3_1', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(64, 3, name='conv3_2', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(64, 3, name='conv3_3', padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.BatchNormalization(name='bn3'),
            layers.Conv2D(128, 3, name='conv4_1', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(128, 3, name='conv4_2', padding='same', kernel_initializer=self.params['kernel_initializer']),
            layers.Conv2D(128, 3, name='conv4_3', padding='same', strides=2, kernel_initializer=self.params['kernel_initializer']),
            layers.Flatten(),
            layers.Dense(128, name='fc5', activation='relu', kernel_initializer=self.params['kernel_initializer']),
            layers.Dropout(self.params['dropout']),
            layers.Dense(10, name='fc6', activation='softmax', kernel_initializer=self.params['kernel_initializer']),
        ])
