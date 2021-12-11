import os
import pandas as pd
import glob


def get_png_path():
    return os.getcwd() + "../IXI-T1-slices/skull-stripped/test"


def augment(image, label):
    # Noise and Dropout
    rnds_noise = tf.random.uniform((1, 2), minval=0, maxval=0.04)
    image = tf.nn.dropout(image, rnds_noise[0][0])
    image = tf.keras.layers.GaussianNoise(rnds_noise[0][1])(image, training=True)

    # Blankout and blur
    rnds_absolutes = tf.random.uniform((1, 2), minval=0, maxval=1)
    if rnds_absolutes[0][0] < 0.2:
        size = tf.random.uniform((), minval=10, maxval=30, dtype=tf.dtypes.int32)
        offset = tf.random.uniform((), minval=10, maxval=100, dtype=tf.dtypes.int32)
        image = tfa.image.cutout(tf.expand_dims(image, 0),
                                 mask_size=(size, size),
                                 offset=(offset, offset),
                                 constant_values=0
                                 )[0, ...]
    if rnds_absolutes[0][1] < 0.1:
        image = tfa.image.gaussian_filter2d(image,
                                            filter_shape=[3, 3],
                                            sigma=0.6,
                                            constant_values=0,
                                            )
    # ZOOM - CROP
    # if rnds_crops[0][1] < 0.1:
    # image = tf.image.central_crop(image, central_fraction=0.7)
    # image = tf.image.resize(image, (128,128))

    # Normalization
    image = tf.math.divide(tf.math.subtract(image, tf.math.reduce_min(image)),
                           tf.math.subtract(tf.math.reduce_max(image), tf.math.reduce_min(image)))
    return image, label


def test():
    # Open a PNG to be augmentated:
    file = get_png_path() + '/' + 'IXI016-Guys-0697-IXI3DMPRAG_-s231_-0301-00003-000001-01.nii.slice-70.ta16498.png';




if __name__ == '__main__':
    test()

