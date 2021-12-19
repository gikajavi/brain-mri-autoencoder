# This is a mix between residual learning and skip connections models
import keras
from keras import layers
from tensorflow.keras.models import Model

class resnet_model():
    def get_model(self, activation='sigmoid', type='pre'):
        input_img = layers.Input(shape=(128, 128, 1))

        res_block = self._pre_activation_residual_block if type == 'pre' else residual_block
        conv2d = layers.Conv2D(32, (3, 3), strides=2, padding="same", name='Conv1')(input_img)
        resblock = res_block(conv2d, 64)
        resblock = res_block(resblock, 64, strides=1)
        encoder = res_block(resblock, 128, strides=2)

        decoder = self._transpose2d(encoder, 64)
        decoder = layers.Concatenate()([decoder, resblock])
        decoder = self._transpose2d(decoder, 32)
        decoder = layers.Concatenate()([decoder, conv2d])
        decoder = self._transpose2d(decoder, 16)
        decoder = layers.Conv2D(1, (3, 3), activation=activation, padding='same')(decoder)

        return Model(input_img, decoder, name="resnet_" + type)

    def _transpose2d(self, inputs, filters):
        block = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(inputs)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)
        return block

    def _residual_block(self, inputs, filters, strides=2):
        """
        See the block diagram:
        https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625
        """
        block = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding="same")(inputs)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)

        block = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same")(block)
        block = layers.BatchNormalization()(block)

        inputs = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding="same")(inputs)

        block = layers.Add()([inputs, block])
        block = layers.ReLU()(block)
        return block

    def _pre_activation_residual_block(self, inputs, filters, strides=2):
        """
        See the block diagram:
        https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625
        "the pre-activation architecture is implemented by moving BN and ReLU activation function before convolution operation"
        """
        block = layers.BatchNormalization()(inputs)
        block = layers.ReLU()(block)

        block = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding="same")(block)

        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)

        block = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="same")(block)

        if strides != 1:
            inputs = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding="same")(inputs)

        block = layers.Add()([inputs, block])
        return block