
# %tensorflow_version 2.x

# install tensorflow 2 and tensorflow datasets on a personal machine
# !pip install tensorflow-gpu
# !pip install tensorflow-datasets

import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from keras.models import Model
from keras import regularizers

import numpy as np



'''
Encoder - Level 0 - special bottleneck - repeat 1x
==================================================

    Input
    -----
       tensor: 28 x 28 x 16

    Residuals
    ---------
       filter: 16 x 1 x 1 x 16
       filter: 16 x 3 x 3 x 16
       filter: 64 x 1 x 1 x 16

    Main path
    ---------
       filter: 64 x 1 x 1 x 16

    Output
    ------
       tensor: 28 x 28 x 64

Encoder - Level 0 - standard bottleneck - repeat (num_0_blocks - 1)x
====================================================================

    Input
    -----
       tensor: 28 x 28 x 64

    Residuals
    ---------
       filter: 16 x 1 x 1 x 64
       filter: 16 x 3 x 3 x 16
       filter: 64 x 1 x 1 x 16

    Main path
    ---------
       filter: identity

    Output
    ------
       tensor: 28 x 28 x 64

Encoder - level 1 - down sampling bottleneck - repeat 1x
========================================================

    Input
    -----
      tensor:  28 x 28 x 64

    Residual path
    -------------
      filter:  32 x 1 x 1 x 64 / 2
      filter:  32 x 3 x 3 x 32
      filter: 128 x 1 x 1 x 32

    Main path
    ---------
      filter: 128 x 1 x 1 x 64 / 2

    Output
    ------
      tensor:  14 x 14 x 128

Encoder - level 1 - standard bottleneck - repeat (num_1_blocks - 1)x
====================================================================
    Input
    -----
       tensor:  14 x 14 x 128

    Residuals
    ---------
       filter:  32 x 1 x 1 x 128
       filter:  32 x 3 x 3 x 32
       filter: 128 x 1 x 1 x 32

    Main path
    ---------
       filter: identity

    Output
    ------
       tensor:  14 x 14 x 128

Encoder - Level 2 - down sampling bottleneck - repeat 1x
========================================================

    Input
    -----
       tensor:  14 x 14 x 128

    Residuals
    ---------
       filter:  64 x 1 x 1 x 128 / 2
       filter:  64 x 3 x 3 x 64
       filter: 256 x 1 x 1 x 64

    Main path
    ---------
       filter: 256 x 1 x 1 x 128 / 2

    Output
    ------
       tensor:   7 x 7 x 256
            
Encoder - Level 2 - standard bottleneck - repeat (num_2_blocks - 1)x
====================================================================

    Input
    -----
       tensor:   7 x 7 x 256

    Residuals
    ---------
       filter:  64 x 1 x 1 x 256
       filter:  64 x 3 x 3 x 64
       filter: 256 x 1 x 1 x 64

    Main path
    ---------
       filter: identity

    Output
    ------
       tensor:   7 x 7 x 256

Encoder - Level 2 - standard bottleneck complete
================================================

    Input
    -----
       tensor:   7 x 7 x 256

    Main path
    ---------
       batch norm
       ReLU

    Output
    ------
       tensor:   7 x 7 x 256
'''


class ResNetv2(Model):


    def __init__(self, num_classes, momentum=.99, epsilon=.001, channels=3, blocks_shape=(4,6,3,)):
        super(ResNetv2, self).__init__()
        self.encoder_tail = Conv2D(16, 3, strides=1, padding='same', activation=None, use_bias=False)
        filters           = 16
        self.blocks       = []
        # Iterate over levels of ResNet.
        for i in range(3):
            self.blocks.append(ResNetv2.ConvBlock(filters, strides=2))
            self.blocks.extend([ResNetv2.ResidualBlock(filters, strides=1) for _ in range(blocks_shape[i])])

            filters *= 2

        self.batch_norm   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
        self.relu         = ReLU()

        self.pool         = GlobalAveragePooling2D()
        self.decoder      = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        # Encoder - Tail.
        x = self.encoder_tail(inputs)
        for layer in self.blocks:
            x = layer(x)

        x = self.batch_norm(x)
        x = self.relu(x)
        # Encoder - Output.
        encoder_output = x
        # Decoder.
        y              = self.pool(encoder_output)
        decoder_output = self.decoder(y) 
        return decoder_output  


    class ConvBlock(keras.layers.Layer):


        def __init__(self, filters, strides=2, momentum=.99, epsilon=.001):
            super(ResNetv2.ConvBlock, self).__init__()
            self.bn1   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu1 = ReLU()
            self.conv1 = Conv2D(filters, 1, strides=strides, padding='same', activation=None, use_bias=False)

            self.bn2   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu2 = ReLU()
            self.conv2 = Conv2D(filters, 3, strides=1, padding='same', activation=None, use_bias=False)

            self.bn3   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu3 = ReLU()
            self.conv3 = Conv2D(4*filters, 1, strides=1, padding='same', activation=None, use_bias=False)

            self.conv4 = Conv2D(4*filters, 1, strides=strides, padding='same', activation=None, use_bias=False)
            self.add   = Add()


        def call(self, inputs):
            residual = self.bn1(inputs)
            residual = self.relu1(residual)
            residual = self.conv1(residual)
            residual = self.bn2(residual)
            residual = self.relu2(residual)
            residual = self.bn3(residual)
            residual = self.relu3(residual)
            residual = self.conv3(residual)

            x        = self.conv4(inputs)
            x        = self.add([x, residual])
            return x


    class ResidualBlock(keras.layers.Layer):


        def __init__(self, filters, strides=1, momentum=.99, epsilon=.001):
            super(ResNetv2.ResidualBlock, self).__init__()
            self.bn1   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu1 = ReLU()
            self.conv1 = Conv2D(filters, 1, strides=strides, padding='same', activation=None, use_bias=False)

            self.bn2   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu2 = ReLU()
            self.conv2 = Conv2D(filters, 3, strides=strides, padding='same', activation=None, use_bias=False)

            self.bn3   = BatchNormalization(axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True)
            self.relu3 = ReLU()
            self.conv3 = Conv2D(4*filters, 1, strides=strides, padding='same', activation=None, use_bias=False)

            self.add   = Add()


        def call(self, inputs):
            residual = self.bn1(inputs)
            residual = self.relu1(residual)
            residual = self.conv1(residual)
            residual = self.bn2(residual)
            residual = self.relu2(residual)
            residual = self.bn3(residual)
            residual = self.relu3(residual)
            residual = self.conv3(residual)

            x        = self.add([inputs, residual])
            return x