"""
|
|
|
|
|
|
|
|
"""
import os

import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from keras import backend as K


class AutoEncoder:
    """
    Deep convolutional autoencoder architecture
    """

    def __init__(self,
                 input_shape,
                 filters,
                 kernels,
                 strides,
                 latent_space_dimension):
        self.input_shape = input_shape  # [28, 28, 2] -- [h, w, #channels]
        self.filters = filters  # [2, 4, 8] -- number of channels for each CL
        self.kernels = kernels  # [3, 5, 3] -- side length of the square kernel for each CL
        self.strides = strides  # [2, 2, 2] -- stride size for each CL
        self.latent_space_dimension = latent_space_dimension  # bottleneck neurons

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(filters)
        self.shape_before_bottleneck = None

        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_auto_encoder()

    def _build_encoder(self):
        encoder_input = self._add_input_layer()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)

        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_input_layer(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """creates convolutional blocks of an encoder"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        adds a convolutional block to a graph of layers consisting of:
        1. Conv2D
        2. ReLU activation
        3. Batch Normalization
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(self.filters[layer_index],
                            self.kernels[layer_index],
                            self.strides[layer_index],
                            padding='same',
                            name=f'encoder_conv_layer_{layer_number}')
        x = conv_layer(x)
        x = ReLU(name=f'encoder_relu_{layer_number}')(x)
        x = BatchNormalization(name=f'encoder_bn_{layer_number}')(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck -- affine (fully connected) layer"""
        self.shape_before_bottleneck = K.int_shape(x)[1:]  # [2(batch), 7(w), 7(h), 32(chans)]
        x = Flatten(name='encoder_flatten')(x)
        x = Dense(self.latent_space_dimension, name='encoder_output')(x)
        return x

    def _build_decoder(self):
        pass

    def _build_auto_encoder(self):
        pass

    def summary(self):
        self.encoder.summary()
        pass


if __name__ == '__main__':
    # autoencoder = AutoEncoder(
    #     input_shape=(28, 28, 1),
    #     filters=(32, 64, 64, 64),
    #     kernels=(3, 3, 3, 3),
    #     strides=(1, 2, 2, 1),
    #     latent_space_dimension=2
    # )
    #
    #
    # autoencoder.summary()

    specs_path = '../data/spectrograms/'

    collector = []
    for root, _, files in os.walk(specs_path):
        for file in files:
            if 'master' in file.lower():
                continue
            if len(collector) > 100:
                break

            temp = np.load(specs_path + file)
            collector.append(temp)


    test_data = np.asarray(collector)
    test_data = test_data.reshape(test_data.shape + (1,))

    autoencoder = AutoEncoder(
        input_shape=test_data.shape[1:],
        filters=(32, 64, 64, 64),
        kernels=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        latent_space_dimension=5
    )
    autoencoder.summary()
    a = autoencoder.encoder(test_data)
    print(a)
