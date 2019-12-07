#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import re
import math
import numpy as np
import tensorflow as tf
from config.param import IS_TRAIN, measure_dict, RANDOM_STATE, NEW_TIME_DIR
from config.path import PATH_MODEL_DIR, PATH_BOARD_DIR, mkdir_time
from tf_callback.Saver import Saver

keras = tf.keras


class NN:
    default_params = {
        'learning_rate': 1e-7,
        'lr_decay_rate': 0.1,
        'lr_staircase': True,
        'batch_size': 5,
        'epoch': 100,
        'early_stop': 30,
        'auto_open_tensorboard': True,
        'monitor': 'val_categorical_accuracy',
        'monitor_mode': 'max',
        'monitor_start_train_acc': 0.65,
        'initial_epoch': 0,
        'random_state': RANDOM_STATE,
    }

    # model param config
    params = {**default_params}

    @property
    def config_for_keras(self):
        """ NEED: Customize the config for keras """
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

    def __init__(self, model_dir, model_name=None):
        self.time_dir = model_dir
        self.__model_dir = mkdir_time(PATH_MODEL_DIR, model_dir)
        self.__monitor_bigger_best = self.params['monitor_mode'] == 'max'

        # initialize some variables that would be used by func "model.fit";
        #   the child class can change this params when customizing the build func
        self.__class_weight = None
        self.__initial_epoch = 0 if 'initial_epoch' not in self.params else self.params['initial_epoch']

        # get the tensorboard dir path
        self.__get_tf_board_path(model_dir)

        # get the model path
        self.__get_model_path(model_name)

        # build model
        self.build()

        # initialize some callback funcs
        self.__init_callback()

    def build(self):
        """ Build neural network architecture; Need to customize """
        self.model = None

    def __get_model_path(self, model_name):
        """ Get the model path """
        self.model_path = os.path.join(self.__model_dir, model_name + '.hdf5')
        self.checkpoint_path = os.path.join(self.__model_dir,
                                            model_name + '.{epoch:03d}-{%s:.4f}.hdf5' % self.params['monitor'])

        self.__new_model_dir = os.path.join(os.path.split(self.__model_dir)[0], NEW_TIME_DIR)
        self.__update_model_path = os.path.join(self.__new_model_dir, model_name + '.hdf5')
        self.__update_checkpoint_path = os.path.join(self.__new_model_dir,
                                                     model_name + '.{epoch:03d}-{%s:.4f}.hdf5' % self.params['monitor'])

        # check if model exists
        if not os.path.isfile(self.model_path) and not os.path.isfile(self.model_path + '.index'):
            model_path = self.__get_best_model_path()
            if model_path:
                self.model_path = model_path

    def test_func(self):
        self.__get_best_model_path()

    def __get_best_model_path(self):
        """ Return the best model in model_dir """
        # check if any model exists
        if not os.listdir(self.__model_dir):
            return

        # initialize some variables
        best = -np.inf if self.__monitor_bigger_best else np.inf
        best_epoch = 0
        best_file_name = ''
        reg = re.compile('\.(\d+)-(\d+\.\d+)\.hdf5')

        # check all the model name in model dir
        for file_name in os.listdir(self.__model_dir):
            # filter irrelevant files
            if '.hdf5' != file_name[-len('.hdf5'):] and '.hdf5.index' != file_name[-len('.hdf5.index'):]:
                continue

            epoch, monitor = reg.findall(file_name)[0]
            epoch = int(epoch)
            monitor = float(monitor)

            # compare the result, find out if it is the best
            if (self.__monitor_bigger_best and best < monitor) or (not self.__monitor_bigger_best and best > monitor) \
                    or (best == monitor and best_epoch < epoch):
                best = monitor
                best_file_name = file_name.replace('.hdf5', '').replace('.index', '')
                best_epoch = epoch

        return os.path.join(self.__model_dir, best_file_name + '.hdf5')

    def __get_tf_board_path(self, model_dir):
        """ Get the tensorboard dir path and run it on cmd """
        self.tf_board_dir = mkdir_time(PATH_BOARD_DIR, model_dir)

    def __init_variables(self, data_size):
        """ Initialize some variables that will be used while training """
        self.__global_step = tf.train.get_or_create_global_step()
        self.__steps_per_epoch = int(data_size // self.params['batch_size'])
        self.__steps = self.__steps_per_epoch * self.params['epoch']

        self.__decay_steps = self.__steps if not self.params['lr_staircase'] else self.__steps_per_epoch

        self.__learning_rate = tf.train.exponential_decay(self.params['learning_rate'], self.__global_step,
                                                          self.__decay_steps, self.params['lr_decay_rate'],
                                                          self.params['lr_staircase'])
        tf.summary.scalar('learning_rate', self.__learning_rate)

    def __init_callback(self):
        """ Customize some callbacks """
        self.callback_tf_board = keras.callbacks.TensorBoard(log_dir=self.tf_board_dir,
                                                             histogram_freq=1,
                                                             write_grads=True,
                                                             write_graph=True,
                                                             write_images=True)
        self.callback_tf_board.set_model(self.model)

        self.callback_saver = Saver(self.checkpoint_path,
                                    self.params['monitor'],
                                    self.params['monitor_mode'],
                                    self.params['early_stop'],
                                    self.params['monitor_start_train_acc'])
        self.callback_saver.set_model(self.model)

    def before_train(self, data_size, x, y):
        self.__init_variables(data_size)

        self.compile(self.__learning_rate)

        # initialize global variables
        keras.backend.get_session().run(tf.global_variables_initializer())

        # if model exists, load the model weight
        if os.path.isfile(self.model_path) or os.path.isfile(self.model_path + '.index'):
            self.load_model(self.model_path, x, y)
            self.model_path = self.__update_model_path
            self.checkpoint_path = self.__update_checkpoint_path

    def train(self, train_x, train_y_one_hot, val_x, val_y_one_hot):
        """ Train model with all data loaded in memory """
        self.before_train(len(train_y_one_hot), train_x, train_y_one_hot)

        if IS_TRAIN:
            # The returned value may be useful in the future
            history_object = self.model.fit(train_x, train_y_one_hot,
                                            epochs=self.params['epoch'],
                                            batch_size=self.params['batch_size'],
                                            validation_data=(val_x, val_y_one_hot),
                                            callbacks=self.config_for_keras['callbacks'],
                                            class_weight=self.__class_weight,
                                            initial_epoch=self.__initial_epoch,
                                            verbose=2)

            best_model_path = self.__get_best_model_path()
            if best_model_path:
                self.load_model(best_model_path)

        return self.test(train_x, train_y_one_hot, val_x, val_y_one_hot)

    @staticmethod
    def reset_graph():
        keras.backend.get_session().close()
        tf.reset_default_graph()
        keras.backend.set_session(tf.Session(graph=tf.get_default_graph()))

    def compile(self, learning_rate):
        self.model.compile(optimizer=self.config_for_keras['optimizer'](learning_rate),
                           loss=self.config_for_keras['loss'],
                           metrics=self.config_for_keras['metrics'])

    def load_model(self, model_path, x=None, y=None):
        # empty fit, to prevent error from occurring when loading model
        self.model.fit(x, y, epochs=0) if not isinstance(x, type(None)) else None

        self.model.load_weights(model_path)
        print('Finish loading weights from %s ' % model_path)

    def test(self, train_x, train_y_one_hot, val_x, val_y_one_hot, name='val'):
        """ Customize for testing model """
        if name == 'train':
            return self.test_in_batch(val_x, val_y_one_hot, name)

        # evaluate the validation data
        return self.measure_and_print(val_y_one_hot, self.predict(val_x), name)

    def test_in_batch(self, x, y_ont_hot, name='val'):
        """ evaluate the model performance while data size is big """
        # variables that record all results
        logits_list = []

        # calculate the total steps
        batch_size = self.params['batch_size']
        steps = int(math.ceil(len(y_ont_hot) * 1.0 / batch_size))

        # traverse all data
        for step in range(steps):
            tmp_x = x[step * batch_size: (step + 1) * batch_size]
            logits_list.append(self.predict(tmp_x))

        logits_list = np.vstack(logits_list)

        return self.measure_and_print(y_ont_hot, logits_list, name)

    @staticmethod
    def measure_and_print(y_ont_hot, logits_list, name='val'):
        result_dict = {}
        for key, func in measure_dict.items():
            result_dict[key] = func(y_ont_hot, logits_list, logits_list[:, 1])

        # show results
        for key, value in result_dict.items():
            print('%s %s: %f' % (name, key, value))

        return result_dict

    def predict(self, x):
        return self.model.predict(x)

    def predict_class(self, x):
        output = self.predict(x)
        return np.argmax(output, axis=-1)

    def predict_correct(self, x, y_one_hot):
        prediction = self.predict_class(x)
        y = np.argmax(y_one_hot, axis=-1)
        return prediction == y

    def predict_prob(self, x, class_index=-1):
        return self.predict(x)[:, class_index]

    def save(self):
        self.model.save_weights(self.model_path, save_format='h5')
        print("Finish saving model to %s " % self.model_path)
