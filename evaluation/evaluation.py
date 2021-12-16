import os
import numpy as np
import glob
import imageio
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def get_random_images(mode="skull-stripped", set='test', n=20):
    """
    returns a randomly sorted list of numpy slices from a subset (test set by default)
    """
    base_dir = f'../input/ixit1slices/IXI-T1-slices/{mode}/{set}'
    files = glob.glob(f'{base_dir}/*.png')
    random.shuffle(files)
    files = files[0:n]
    imgs = []
    for file in files:
        im = imageio.imread(file)
        # prepare data with the same shape the NNs were fed
        im = im.astype(np.float32)
        im = im / 255
        im = im.reshape(256, 256, 1)
        imgs.append(im)

    imgs = tf.image.resize(imgs, [128, 128]).numpy()
    return imgs


def show_img_grid(imgs, ncols=5, show_axis=False):
    nrows = math.ceil(len(imgs) / ncols)
    f, axarr = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    i = 0

    for img in imgs:
        cur_row = i // ncols
        cur_col = i % ncols
        if nrows == 1:
            axarr[cur_col].imshow(np.rot90(img), cmap="gray")
            if not show_axis:
                axarr[cur_col].axis('off')
        else:
            axarr[cur_row, cur_col].imshow(np.rot90(img), cmap="gray")
            if not show_axis:
                axarr[cur_row, cur_col].axis('off')
        i += 1
    plt.show()


def predict_imgs(inputs, model):
    """
    predicts the input images and returns a list with pairs of images in consecutive order
    to easily plot them
    """
    predicted = model.predict(inputs)
    outputs = [None] * (len(inputs) * 2)
    outputs[::2] = inputs
    outputs[1::2] = predicted
    return outputs


# to run in Kaggle notebook:
# imgs = get_random_images(n=8)
# show_img_grid(imgs, ncols=4)
# from keras.models import load_model
# path_to_model = "../input/newgenerator/model_baseline_10epoch_es4_activation_sigmoid.h5"
# model = load_model(path_to_model)
# outputs = predict_imgs(imgs, model)
# show_img_grid(outputs, ncols=4)
