import tensorflow as tf
import keras
from keras import layers


def baseline_skconn_model():
    """
    A model like the baseline but with skip connections
    :return:
    """
    input_img = keras.Input(shape=(128, 128, 1))

    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", name='Conv_1')(input_img)
    bn1 = layers.BatchNormalization()(x)
    x = layers.ReLU()(bn1)

    x = layers.Conv2D(64, (3,3), strides=2, padding="same", name='Conv_2')(x)
    bn2 = layers.BatchNormalization()(x)
    x = layers.ReLU()(bn2)

    x = layers.Conv2D(128, (3,3), strides=2, padding="same", name='Conv_3')(x)
    bn3 = layers.BatchNormalization()(x)
    encoded = layers.ReLU(name='middle_layer')(bn3)

    x = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='Conv2DT_1')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Add(name='sconn1')([x, bn2])
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='Conv2DT_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add(name='sconn2')([x, bn1])
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', name='Conv2DT_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    decoded = layers.Conv2D(1, (3,3), activation='linear', padding='same', name='decoder')(x)

    autoencoder = keras.Model(input_img, decoded, name="baseline_with_skconn")
    return autoencoder



