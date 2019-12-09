""" ResNet18 """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf

layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
    """_IdentityBlock is the block that has no conv layer at shortcut.

    Args:
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
    """

    def __init__(self, filters, stage, block, data_format,
                 kernel_initializer='glorot_uniform', kernel_regularizer=None):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else -1

        self.conv2a = layers.Conv2D(
            filters1, 3, padding='same', name=conv_name_base + '2a', data_format=data_format,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2, 3, padding='same', name=conv_name_base + '2b', data_format=data_format,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    """_ConvBlock is the block that has a conv layer at shortcut.

    Args:
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
          'channels_last').
        strides: strides for the convolution. Note that from stage 3, the first
         conv layer at main path is with strides=(2,2), and the shortcut should
         have strides=(2,2) as well.
    """

    def __init__(self,
                 filters,
                 stage,
                 block,
                 data_format,
                 strides=(2, 2),
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else -1

        self.conv2a = layers.Conv2D(
            filters1, 3, strides=strides, padding='same', name=conv_name_base + '2a', data_format=data_format,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2, 3, padding='same', name=conv_name_base + '2b', data_format=data_format,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv_shortcut = layers.Conv2D(
            filters2, 1, strides=strides, padding='same', name=conv_name_base + '1', data_format=data_format,
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        self.bn_shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


# pylint: disable=not-callable
class ResNet18(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Args:
      data_format: format for the image. Either 'channels_first' or
        'channels_last'.  'channels_first' is typically faster on GPUs while
        'channels_last' is typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
      name: Prefix applied to names of variables created in the model.
      trainable: Is the model trainable? If true, performs backward
          and optimization after call() method.
      include_top: whether to include the fully-connected layer at the top of the
        network.
      pooling: Optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor
            output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of
            the last convolutional layer, and thus the output of the model will be
            a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True.

    Raises:
        ValueError: in case of invalid argument for data_format.
    """

    def __init__(self,
                 data_format,
                 name='',
                 trainable=True,
                 include_top=True,
                 pooling=None,
                 classes=10,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 dropout=None):
        super(ResNet18, self).__init__(name=name)

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))
        self.include_top = include_top

        def conv_block(filters, stage, block, strides=(2, 2)):
            """ :rtype: class """
            return _ConvBlock(filters, stage=stage, block=block, data_format=data_format, strides=strides,
                              kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

        def id_block(filters, stage, block):
            """ :rtype: class """
            return _IdentityBlock(filters, stage=stage, block=block, data_format=data_format,
                                  kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)

        self.conv1 = layers.Conv2D(
            64, (7, 7), strides=(2, 2), data_format=data_format, padding='same', name='conv1',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        bn_axis = 1 if data_format == 'channels_first' else -1
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D(
            (3, 3), strides=(2, 2), data_format=data_format)

        self.l2a = conv_block([64, 64], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64], stage=2, block='b')

        self.l3a = conv_block([128, 128], stage=3, block='a')
        self.l3b = id_block([128, 128], stage=3, block='b')

        self.l4a = conv_block([256, 256], stage=4, block='a')
        self.l4b = id_block([256, 256], stage=4, block='b')

        self.l5a = conv_block([512, 512], stage=5, block='a')
        self.l5b = id_block([512, 512], stage=5, block='b')

        # self.avg_pool = layers.AveragePooling2D(
        #     (2, 2), strides=(2, 2), data_format=data_format)

        if self.include_top:
            self.flatten = layers.Flatten()
            self.dropout = layers.Dropout(dropout) if dropout else None
            self.fc10 = layers.Dense(classes, name='fc10', activation='softmax')
        else:
            reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
            reduction_indices = tf.constant(reduction_indices)
            if pooling == 'avg':
                self.global_pooling = functools.partial(
                    tf.reduce_mean,
                    reduction_indices=reduction_indices,
                    keep_dims=False)
            elif pooling == 'max':
                self.global_pooling = functools.partial(
                    tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
            else:
                self.global_pooling = None

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)

        # x = self.avg_pool(x)

        if self.include_top:
            x = self.flatten(x)
            if not isinstance(self.dropout, type(None)):
                x = self.dropout(x, training=training)
            return self.fc10(x)
        elif self.global_pooling:
            return self.global_pooling(x)
        else:
            return x