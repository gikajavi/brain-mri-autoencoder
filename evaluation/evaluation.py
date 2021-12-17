import os
import numpy as np
import pandas as pd
import glob
import imageio
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from keras import layers
import tensorflow_addons as tfa
import pickle

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.precision', 3)


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


def predict_imgs(inputs, model, augmentation=[]):
    """
    predicts the input images and returns a list with pairs of images in consecutive order
    to easily plot them
    """
    if len(augmentation) > 0:
        augment = Augmentation()
        augmented = inputs
        if "noise" in augmentation:
            augmented = augment.add_noise(augmented)
        if "blur" in augmentation:
            augmented = augment.gaussian_blur(augmented)
        if "dropout" in augmentation:
            augmented = augment.dropout(augmented)
        if "cutout" in augmentation:
            augmented = augment.cutout(augmented)
        if "random" in augmentation:
            augmented = augment.random(augmented)
        predicted = model.predict(augmented)
        outputs = [None] * (len(inputs) * 3)
        outputs[::3] = inputs
        outputs[1::3] = augmented
        outputs[2::3] = predicted
        return outputs
    else:
        predicted = model.predict(inputs)
        outputs = [None] * (len(inputs) * 2)
        outputs[::2] = inputs
        outputs[1::2] = predicted
        return outputs


# Els historials van ser emmagatzemats al finalitzar cada entrenament en diferents sessions de Kaggle.
# Ara en recuperem alguns per mostrar les gràfiques.
def show_history_data(path, show_table=True, plot=True):
    with open(path, "rb") as input_file:
        history = pickle.load(input_file)

    if show_table:
        l = []
        for i in range(0, len(history['loss'])):
            l.append({'Epoch': i + 1, 'loss': history['loss'][i], 'val_loss': history['val_loss'][i]})
        print(pd.DataFrame(l))

    if plot:
        plt.style.use("ggplot")
        plt.plot(np.arange(0, len(history['loss'])), history["loss"], label="train")
        plt.plot(np.arange(0, len(history['val_loss'])), history["val_loss"], label="val")
        plt.title("Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.show()


# to run in Kaggle notebook:
# imgs = get_random_images(n=8)
# show_img_grid(imgs, ncols=4)
# from keras.models import load_model
# path_to_model = "../input/newgenerator/model_baseline_10epoch_es4_activation_sigmoid.h5"
# model = load_model(path_to_model)
# outputs = predict_imgs(imgs, model)
# show_img_grid(outputs, ncols=4)


# Testing model trained with full-skull images, during 25 epochs, activation linear
# from keras.models import load_model
# path_to_model = "../input/fullskullresnet-sconn-25epoch-es5-activa-linear/full-skull-resnet_sconn_25epoch_es5_activation_linear/full-skull-resnet_sconn_25epoch_es5_activation_linear.h5"
# model = load_model(path_to_model)
# outputs = predict_imgs(imgs, model, ['blur'])
# show_img_grid(outputs, ncols=3)


# Load and show history data
# path = '../input/fullskullresnet-sconn-25epoch-es5-activa-linear/full-skull-resnet_sconn_25epoch_es5_activation_linear/full-skull-resnet_sconn_25epoch_es5_activation_linear.history'
# show_history_data(path)
# path = '../input/newgenerator/model_baseline_10epoch_es20_activation_sigmoid.h12.history'
# show_history_data(path)