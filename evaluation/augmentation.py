import random
import tensorflow as tf
from keras import layers
import tensorflow_addons as tfa

class Augmentation():
    """
    Helper to apply some filters to images
    """
    def random(self, images):
        # Each filter is applied with a probability of 25%, except cutout which is applied half of the times
        if random.randint(1, 4) == 1:
            images = self.add_noise(images)
        if random.randint(1, 4) == 1:
            images = self.dropout(images)
        if random.randint(1, 4) == 1:
            images = self.gaussian_blur(images)
        if random.randint(1, 1) == 1:
            images = self.cutout(images)
        return images

    def add_noise(self, images):
        sdev = 0 + (random.random() * (0.04 - 0))
        images = layers.GaussianNoise(stddev=sdev)(images, training=True)
        return images

    def dropout(self, images):
        rnds_noise = tf.random.uniform((1, 2), minval=0, maxval=0.04)
        images = tf.nn.dropout(images, rnds_noise[0][0])
        return images

    # https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
    def gaussian_blur(self, images):
        images = tfa.image.gaussian_filter2d(images,
                                             filter_shape=[4, 4],
                                             sigma=0.8,
                                             constant_values=0,
                                             )
        return images

    def cutout(self, images):
        w = tf.random.uniform((), minval=10, maxval=20, dtype=tf.dtypes.int32)
        h = tf.random.uniform((), minval=10, maxval=20, dtype=tf.dtypes.int32)
        x = tf.random.uniform((), minval=20, maxval=105, dtype=tf.dtypes.int32)
        y = tf.random.uniform((), minval=40, maxval=105, dtype=tf.dtypes.int32)

        if w % 2 != 0:
            w += 1 if bool(random.getrandbits(1)) else -1
        if h % 2 != 0:
            h += 1 if bool(random.getrandbits(1)) else -1

        # image = tfa.image.random_cutout(image, mask_size=(w,h), constant_values=0)
        images = tfa.image.cutout(images,
                                  mask_size=(w, h),
                                  offset=(x, y),
                                  constant_values=0
                                  )
        return images