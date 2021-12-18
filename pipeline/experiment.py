import traceback
import tensorflow as tf
import keras
import pickle
from dataset import DataGenerator


class Experiment:
    _history = None

    def __init__(self, name='Exp', model: keras.Model = None, da_enabled=True, aug_policy='batch', train_gen=None, val_gen=None,
                 skull_stripped=True, path_to_dataset='', path_to_results='', epochs=25, es_patience=5,
                 batch_size=128, optimizer="Adam", loss="mse", metrics="mse", reduce_lr_on_plateau=True):
        self.name = name
        self.model = model
        self.da_enabled = da_enabled
        self.aug_policy = aug_policy
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.skull_stripped = skull_stripped
        self.path_to_dataset = path_to_dataset
        self.path_to_results = path_to_results
        self.epochs = epochs
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.reduce_lr_on_plateau = reduce_lr_on_plateau

        if self.path_to_dataset == '':
            self.path_to_dataset = '../input/ixit1slices/IXI-T1-slices'
        if self.path_to_results == '':
            self.path_to_results = './'

        path_to_slices = '/skull-stripped' if self.skull_stripped else '/full'
        path_to_slices = self.path_to_dataset + path_to_slices
        path_to_train_slices = path_to_slices + '/train'
        path_to_val_slices = path_to_slices + '/val'

        self.train_gen = DataGenerator(base_dir=path_to_train_slices, batch_size=self.batch_size,
                                       Augment=self.da_enabled, AugmentationPolicy=self.aug_policy)

        self.val_gen = DataGenerator(base_dir=path_to_val_slices, batch_size=self.batch_size,
                                     Augment=self.da_enabled, AugmentationPolicy=self.aug_policy)

    def get_name(self):
        """
        The name of this experiment, which depends on the model's name and some of the experiment parameters
        :return:
        """
        da_config: str = 'da-yes' if self.da_enabled else 'da-no'
        skull_mode: str = 'skull-stripped' if self.skull_stripped else 'full-skull'
        rlronplateru = 'rlrop-Y' if self.reduce_lr_on_plateau else 'rlrop-N'
        return f'{self.name}_model-{self.model.name}_{skull_mode}_ep-{self.epochs}_bs-{self.batch_size}_{da_config}_loss-{self.loss}_{rlronplateru}'

    def start(self):
        """
        starts the experirment with the given params
        :return:
        """
        try:
            print('Experiment started')
            if self.model is None:
                raise Exception("No Model was provided")

            print("Model is about to start training")
            self.fit_model()

            print("Training has finished. Saving results.")
            self.save_results()
        except Exception as ex:
            print("General error executing the experiment: " + str(ex))
            print(traceback.format_exc())

    def fit_model(self):
        self.compile_model()

        # Callbacks
        callbacks = []
        callbacks.append(tf.keras.callbacks.TensorBoard(f"logs/{self.get_name()}"))
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=self.es_patience,
                                                       verbose=1, restore_best_weights=True))
        if self.reduce_lr_on_plateau:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                                                  patience=2, verbose=1, cooldown=1))

        self._history = self.model.fit(self.train_gen,
                                       epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       validation_data=self.val_gen,
                                       callbacks=callbacks)

    def compile_model(self):
        loss_func = self.get_loss_function()
        self.model.compile(loss=loss_func, optimizer=self.optimizer, metrics=self.metrics)

    def get_loss_function(self):
        if self.loss == 'ssim':
            return self._loss_ssim
        elif self.loss == 'ms-ssim':
            return self._loss_ms_ssim
        elif self.loss == 'combined':
            return self._loss_mae_ssim
        else:
            return 'mse'

    def _loss_ssim(self, img, pred_img):
        return 1 - tf.reduce_mean(tf.image.ssim(pred_img, img, 1.0))

    def _loss_ms_ssim(self, img, pred_img):
        return 1 - tf.reduce_mean(tf.image.ssim_multiscale(pred_img, img, 1.0))

    def _loss_mae_ssim(self, img, pred_img):
        # https://arxiv.org/pdf/1511.08861.pdf
        # TODO
        return None


    def save_results(self):
        filename = self.path_to_results + self.get_name() + '.h5'
        self.model.save(filename)

        filename = self.path_to_results + self.get_name() + '.history'
        with open(filename, 'wb') as file_pi:
            pickle.dump(self._history.history, file_pi)
