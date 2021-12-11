import pandas as pd
import traceback
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import cv2
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from random import randint


class DataAugmentation:
    enabled = True
    add_noise_prob = 0.05
    dropout_prob = 0.05
    gaussian_blur_prob = 0.05
    cutout_prob = 0.05
    # ImageDataGenerator built-in methods. We discard the ones not listed here.
    rotation_range = 12,
    width_shift_range = 0.3
    height_shift_range = 0.3

    def __init__(self):
        self.enabled = True

    def augment_image(self, image):
        if not self.enabled:
            return image
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
        rnds_noise = tf.random.uniform((1, 2), minval=0, maxval=0.04)
        image = tf.keras.layers.GaussianNoise(rnds_noise[0][1])(image, training=True)
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
        image = image.numpy()
        x_start = randint(0, 108)
        y_start = randint(0, 108)
        w = randint(10, 20)
        h = randint(10, 20)
        x_end = x_start + w
        y_end = y_start + h
        image = cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
        image = tf.convert_to_tensor(image)
        return image


class DataProvider:
    """
    This class is basically a wrapper of keras ImageDataGenerator class, implementing a preprocess function
    for adding some extra filters to the pipeline
    """
    da: DataAugmentation = None
    path_to_dataset = None
    batch_size = None
    _train_generator: ImageDataGenerator = None
    _val_generator: ImageDataGenerator = None

    def get_train_generator(self):
        self._train_generator = ImageDataGenerator(
                        rescale=1.0/255.0,
                        preprocessing_function=self._preprocess_img
                    )
        return self._train_generator.flow_from_directory(
            directory=self.path_to_dataset,
            color_mode="grayscale",
            target_size=(128, 128),
            batch_size=self.batch_size,
            class_mode='input',
            classes=['train'],
            shuffle=True
        )

    def _preprocess_img(self, image):
        image = tf.image.resize(image, [128, 128])
        image = self.da.augment_image(image)
        return image

    def get_val_generator(self):
        self._val_generator = ImageDataGenerator(
                        rescale=1.0/255.0,
                        preprocessing_function=self._preprocess_img_val
                    )
        return self._val_generator.flow_from_directory(
            directory=self.path_to_dataset,
            color_mode="grayscale",
            target_size=(128, 128),
            batch_size=self.batch_size,
            class_mode='input',
            classes=['val'],
            shuffle=True
        )

    def _preprocess_img_val(self, image):
        """
        In validation set, the only operation to perform is resizing image
        :param image:
        :return:
        """
        resized_image = tf.image.resize(image, [128, 128])
        return resized_image