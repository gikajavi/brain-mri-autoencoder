import traceback
import tensorflow as tf
import keras
import pickle
from dataset import DataGenerator

class Experiment:
    _history = None

    def __init__(self, name='Experiment', model: keras.Model = None, da_enabled=True, train_gen=None, val_gen=None,
                 skull_stripped=True, path_to_dataset='', path_to_results='', epochs=25, es_patience=5,
                 batch_size=128, optimizer="Adam", loss="mse", metrics="mse"):
        self.name = name
        self.model = model
        self.da_enabled = da_enabled
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.skull_stripped = skull_stripped
        self.path_to_dataset = path_to_dataset
        self.path_to_results = path_to_results,
        self.epochs = epochs
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        if self.path_to_dataset == '':
            self.path_to_dataset = '../input/ixit1slices/IXI-T1-slices'
        if self.path_to_results == '':
            self.path_to_results = './'

        path_to_slices = '/skull-stripped' if self.skull_stripped else '/full'
        path_to_slices = self.path_to_dataset + path_to_slices
        path_to_train_slices = path_to_slices + '/train'
        path_to_val_slices = path_to_slices + '/val'

        self.train_gen = DataGenerator(base_dir=path_to_train_slices, batch_size=self.batch_size,
                                       Augment=self.da_enabled)

        self.val_gen = DataGenerator(base_dir=path_to_val_slices, batch_size=self.batch_size,
                                     Augment=self.da_enabled)


    def get_name(self):
        """
        The name of this experiment, which depends on the model's name and some of the experiment parameters
        :return:
        """
        da_config: str = 'with_da' if self.da_enabled else 'NO_da'
        return f'{self.name}_{self.model.name}_ep{self.epochs}_bs{self.batch_size}_{da_config}'

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

        self._history = self.model.fit(self.train_gen,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=self.val_gen,
                                  callbacks=[tensorboard_callback, es])


    def save_results(self):
        filename = self.path_to_results + self.get_name() + '.h5'
        self.model.save(filename)

        filename = self.path_to_results + self.get_name() + '.history'
        with open(filename, 'wb') as file_pi:
            pickle.dump(self._history.history, file_pi)
