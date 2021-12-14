from tensorflow import keras
from keras import layers
K = keras.backend
from keras.layers import Layer
import tensorflow as tf
from random import randint
from random import random
from random import getrandbits
import cv2
import tensorflow_addons as tfa


class DaLayer(Layer):
    """
    Custom Keras layer to perform some extra Data Augmentation operatoins
    This layer should be inserted at the beginning of the models, just after the input layer
    It's constrained to this project and assumes (128,128,1) as input shape
    """
    def __init__(self, **kwargs):
        super(DaLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        if training:
            inputs = self.augment_image(inputs)

        return inputs

    def augment_image(self, image):
        # We apply each filter 20% of the times, so, since we use up to 4 filters, 20% of times
        # no filters will be applied
        if randint(1, 5) == 1:
            image = self.add_noise(image)
        if randint(1, 5) == 1:
            image = self.dropout(image)
        if randint(1, 5) == 1:
            image = self.gaussian_blur(image)
        if randint(1, 5) == 1:
            image = self.cutout(image)
        return image

    def add_noise(self, image):
        sdev = 0 + (random() * (0.05 - 0))
        image = layers.GaussianNoise(stddev=sdev)(image, training=True)
        return image

    def dropout(self, image):
        rnds_noise = tf.random.uniform((1, 2), minval=0, maxval=0.04)
        image = tf.nn.dropout(image, rnds_noise[0][0])
        return image

    # https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
    def gaussian_blur(self, image):
        image = tfa.image.gaussian_filter2d(image,
                                            filter_shape=[4, 4],
                                            sigma=0.8,
                                            constant_values=0,
                                            )
        return image

    def cutout(self, image):
        w = tf.random.uniform((), minval=10, maxval=20, dtype=tf.dtypes.int32)
        h = tf.random.uniform((), minval=10, maxval=20, dtype=tf.dtypes.int32)
        x = tf.random.uniform((), minval=20, maxval=105, dtype=tf.dtypes.int32)
        y = tf.random.uniform((), minval=40, maxval=105, dtype=tf.dtypes.int32)

        if w % 2 != 0:
            w += 1 if bool(getrandbits(1)) else -1
        if h % 2 != 0:
            h += 1 if bool(getrandbits(1)) else -1

        print(w,h,x,y)
        image = tfa.image.cutout(tf.expand_dims(image, 0),
                                 mask_size=(w, h),
                                 offset=(x, y),
                                 constant_values=0
                                 )[0, ...]
        return image
