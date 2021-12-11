import os
import pandas as pd
import glob
import traceback
import tensorflow as tf
import tensorflow_addons as tfa
import keras
import cv2
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from random import randint
import pickle
from dataset import DataAugmentation
from dataset import DataProvider


class Experiment:
    name = ''
    model: keras.Model = None
    da: DataAugmentation = None
    da_enabled = True
    _data_provider: DataProvider = None
    skull_stripped = True
    path_to_dataset = None
    path_to_results = None
    epochs = 50
    es_patience = 5
    batch_size = 128
    optimizer = "Adam"
    loss = "mse"
    metrics = "accuracy"
    _history = None

    def __init__(self):
        self.name = 'Experiment'
        self.da = DataAugmentation()
        self.da.enabled = self.da_enabled
        # Paths assuming a kaggle environment
        self.path_to_dataset = '../input/ixit1slices/IXI-T1-slices'
        self.path_to_results = './'

    def get_data_provider(self):
        if self._data_provider is None:
            self._data_provider = DataProvider()
            self._data_provider.batch_size = self.batch_size
        return self._data_provider

    def get_name(self):
        """
        The name of this experiment, which depends on the model's name and some of the experiment parameters
        :return:
        """
        da_config: str = 'with_da' if self._data_provider.da.enabled else 'NO_da'
        return f'{self.model.name}_ep{self.epochs}_bs{self.batch_size}_{da_config}'

    def start(self):
        """
        starts the experirment with the given params
        :return:
        """
        try:
            print('Experiment started')
            if self.model is None:
                raise Exception("No Model was provided")

            print("Configuring data paths and augmentation")
            data_provider = self.get_data_provider()
            data_provider.da = self.da
            path_to_slices = '/skull-stripped' if self.skull_stripped else '/full'
            data_provider.path_to_dataset = self.path_to_dataset + path_to_slices

            print("Model is about to start training")
            self.fit_model()

            print("Trainig has finished. Saving results.")
            self.save_results()
        except Exception as ex:
            print("General error executing the experiment: " + str(ex))
            print(traceback.format_exc())


    def fit_model(self):
        # TODO: params should be configurable. accuracy does not make sense for this tasks. We should
        # use SSIM or something more appropriate
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # Callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(f"logs/{self.get_name()}")
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.es_patience, verbose=1,
                                           restore_best_weights=True)

        self._history = self.model.fit(self._data_provider.get_train_generator(),
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=self._data_provider.get_val_generator(),
                                  callbacks=[tensorboard_callback, es])


    def save_results(self):
        filename = self.path_to_results + self.get_name() + '.h5'
        self.model.save(filename)

        filename = self.path_to_results + self.get_name() + '.history'
        with open(filename, 'wb') as file_pi:
            pickle.dump(self._history.history, file_pi)
