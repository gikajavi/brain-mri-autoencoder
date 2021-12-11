import tensorflow as tf
import keras
from keras import layers

def baseline_model():
    input_img = keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)

    # This layer contains the latent space:
    encoded = layers.MaxPooling2D((2, 2), name='middle_layer')(x)

    x = layers.Conv2DTranspose(192, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2DTranspose(96, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(48, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded, name="baseline")
    return autoencoder