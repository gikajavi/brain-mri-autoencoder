import numpy as np
import glob
import traceback
import tensorflow as tf
import tensorflow_addons as tfa
import random
import keras
from keras import layers
import imageio


class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom class to provide data to keras models
    AugmentationPolicy: can be => 'instance', 'batch'
    """
    def __init__(self, base_dir='.', batch_size=128, Shuffle=True, Augment=True, AugmentationPolicy='instance',
                 brain_amount=15):
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.shuffle = Shuffle
        self.Augment = Augment
        self.AugmentationPolicy = AugmentationPolicy
        self.brain_amount = brain_amount

        self.files = glob.glob(f'{base_dir}/*.png')
        if self.brain_amount is not None:
            self._filter_according_to_bamount()

        if self.shuffle:
            random.shuffle(self.files)
        print(f'Data generator on {base_dir}, found {len(self.files)} PNG files')


    def _filter_according_to_bamount(self):
        filtered = []
        for file in self.files:
            ta = int(file.split('.')[-2].replace('ta', ''))
            # ta is the number of pixels belonging to brain tissue in the original 256x256 slice
            percentage = ta/(256*256)*100
            if percentage > self.brain_amount:
                filtered.append(file)
        self.files = filtered

    def __len__(self):
        return len(self.files) // self.batch_size
        # return math.ceil(len(self.files) / self.batch_size)

    def __getitem__(self, index):
        Y = []
        for i in range(index * self.batch_size, index * self.batch_size + self.batch_size):
            image_path = self.files[i]
            im = imageio.imread(image_path)
            im = im.astype(np.float32)
            im = im / 255
            im = im.reshape(256, 256, 1)
            im = tf.image.resize(im, [128, 128])
            Y.append(im)

        # Convert Y to numpy before performing augmentation
        Y = np.array(Y)

        # Augmentation according to policy
        if self.Augment:
            if self.AugmentationPolicy == 'instance':
                X = []
                for i in range(0, len(Y)):
                    augmented_y = self._augment(np.array([Y[i]]))
                    X.append(augmented_y[0])
                X = np.array(X)
            else:
                # All the images in batch will have the same augmentations
                X = Y
                X = self._augment(X)

        return X, Y

    #     def on_epoch_end(self):
    #         self.indexes = np.arange(len(self.list_IDs))
    #         if self.shuffle == True:
    #             np.random.shuffle(self.indexes)

    def _augment(self, images):
        # Each filter is applied with a probability of 25%, except cutout which is applied half of the times
        if random.randint(1, 4) == 1:
            images = self.add_noise(images)
        if random.randint(1, 4) == 1:
            images = self.dropout(images)
        if random.randint(1, 4) == 1:
            images = self.gaussian_blur(images)
        if random.randint(1, 2) == 1:
            images = self.cutout(images)
        return images

    def add_noise(self, images):
        sdev = 0 + (random.random() * (0.05 - 0))
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
