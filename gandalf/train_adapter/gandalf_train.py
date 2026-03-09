from jsonschema import validate, ValidationError
import tensorflow as tf
from pickle import dump
import numpy as np
import datetime
import yaml
import json
import sys
import os

# My modules
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
from gandalf.train_adapter.utils.log_utils import generate_new_result_row
import gandalf.train_adapter.utils.train_utils as train_utils
from gandalf.train_adapter.gandalf_train_interface import GandalfTrainInterface
from gandalf.train.data_preparation import DataLoaderFactory
import gandalf.train.models as my_models

SCHEMA = "/schemas/config.json"

class GandalfTrain(GandalfTrainInterface):
    ''' Class to define and train disentangling models using GANDALF '''

    def __init__(self, config):
        super().__init__(config)

        # validate config
        with open(os.path.dirname(__file__) + SCHEMA, 'r') as f:
            schema = json.load(f)
        try:
            validate(instance=self._config, schema=schema)
        except ValidationError as e:
            print(e)
            sys.exit(1)

        # Take main parameters and data, models and training parameters (dics)
        self.model_id = self._config['id']
        self.root_folder = self._config['root_folder']
        data_config = self._config['data']
        models_config = self._config['models']
        training_config = self._config['train']

        # Data parameters
        self.data_loader = data_config['data_loader']
        self.survey_data_path = data_config['survey_data_path']
        self.labels = data_config['labels']
        self.cond_labels = data_config['cond_labels']
        self.normalize_spectra = data_config['normalize_spectra']
        self.shuffle = data_config['shuffle']
        self.discretize = data_config['discretize']
        self.nbins = data_config['nbins']
        self.seed = data_config['seed']

        # Model parameters
        self.latent_size = models_config['latent_size']
        self.encoder_hidden_layer_sizes = models_config['encoder_hidden_layer_sizes']
        self.decoder_hidden_layer_sizes = models_config['decoder_hidden_layer_sizes']
        self.input_without_params = models_config['input_without_params']
        self.batch_norm = models_config['batch_norm']
        self.disc_hidden_layer_sizes = models_config['disc_hidden_layer_sizes']
        self.conv_disc = models_config['conv_disc']
        self.multi_disc = models_config['multi_disc']

        # Training parameters
        self.epochs = training_config['epochs']
        self.batch_size = training_config['batch_size']
        self.lambda_values = training_config['lambda_values']
        self.lr_disc = training_config['lr_disc']
        self.lr_ae = training_config['lr_ae']
        self.dl_no_change = training_config['dl_no_change']
        self.dl_lambda_steps = training_config['dl_lambda_steps']

        # Private attributes
        self._dataloader = None
        self._data_dim = None
        self._output_discriminator = None

    @classmethod
    def from_config_file(cls, config_file, file_type='yaml'):
        # Load configuration file
        if file_type == 'yaml':
            config = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)
        else:
            raise ValueError('Config file type not supported')

        return cls(config)

    def create_datasets(self, verbose=0):

        self._dataloader = DataLoaderFactory().create_dataloader(
            self.data_loader, self.survey_data_path, self.labels, self.cond_labels,
            self.normalize_spectra, self.shuffle, self.discretize, self.nbins,
            self.seed, verbose)

        dataset_train, dataset_test = self._dataloader.create_tf_dataset(batch=self.batch_size,
                                                            multi_discriminator=self.multi_disc,
                                                            verbose=0) # the verbose of this function is not useful

        # Get data dimension and output discriminator for model building
        self._data_dim = dataset_train.element_spec[0].shape[1]
        self._output_discriminator = dataset_train.element_spec[2].shape[1] if self.discretize else len(self.cond_labels)

        # Save X_train_shape and X_test_shape shape for results
        len_train = len(self._dataloader.get_ids_train())
        len_test = len(self._dataloader.get_ids_test())
        self._X_train_shape = (len_train, self._data_dim)
        self._X_test_shape = (len_test, self._data_dim)

        if verbose:
            print("\nTrain length: ", len_train)
            print("Test length: ", len_test)

            print("\nNumber of parameters: ", len(self.labels))
            print("Number of conditioned parameters: ", len(self.cond_labels))

            print('\nData dimension: {:}'.format(self._data_dim))
            print('Output discriminator: {:}'.format(self._output_discriminator))

        return dataset_train, dataset_test

    def create_autoencoder(self, summary_models=False):
        print('Building Autoencoder')

        # If encoder/decoder are not initialized, the 'make_autoencoder_model' function creates them with the default parameters
        encoder = my_models.make_encoder_model(data_dim=self._data_dim,
                                            params_dim=len(self.cond_labels),
                                            latent_dim=self.latent_size,
                                            hidden_layer_sizes=self.encoder_hidden_layer_sizes,
                                            hidden_layers_activation='relu',
                                            input_without_params=self.input_without_params,
                                            batch_norm=self.batch_norm,
                                            verbose=summary_models)

        decoder = my_models.make_decoder_model(data_dim=self._data_dim,
                                            params_dim=len(self.cond_labels),
                                            latent_dim=self.latent_size,
                                            hidden_layer_sizes=self.decoder_hidden_layer_sizes,
                                            hidden_layers_activation='relu',
                                            batch_norm=self.batch_norm,
                                            verbose=summary_models)



        autoencoder = my_models.make_autoencoder_model(data_dim=self._data_dim, params_dim=len(self.cond_labels),
                                                    latent_dim=self.latent_size,
                                                    encoder=encoder,
                                                    decoder=decoder,
                                                    input_without_params=self.input_without_params)

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder

        print('Done.')

    def create_discriminator(self, summary_models=False):
        print('Building discriminator')

        disc_models_dict = dict()

        if self.multi_disc:
            for y_cond in self.cond_labels:
                _discriminator = my_models.make_discriminator(latent_dim=self.latent_size, nbins=self._output_discriminator,
                                                        hidden_layer_sizes=self.disc_hidden_layer_sizes,
                                                        convolutional=self.conv_disc,
                                                        batch_norm=self.batch_norm,
                                                        output_activation='sigmoid' if self.discretize else 'linear',
                                                        verbose=summary_models)
                disc_models_dict[y_cond+'_discriminator'] = _discriminator
        else:
            _discriminator = my_models.make_discriminator(latent_dim=self.latent_size, nbins=self._output_discriminator,
                                                    hidden_layer_sizes=self.disc_hidden_layer_sizes,
                                                    convolutional=self.conv_disc,
                                                    batch_norm=self.batch_norm,
                                                    output_activation='sigmoid' if self.discretize else 'linear',
                                                    verbose=summary_models)
            disc_models_dict['_'.join(self.cond_labels)+'_discriminator'] = _discriminator

        self.disc_models_dict = disc_models_dict
        print('Done.')

    def train(self, dataset, search_ckpt=False, ckpt_step=10, saved_model_path='results/models', verbose=1):

        # Generate id (if not passed by parameter) for the logs, checkpoint and final models
        if self.model_id is None:
            self.model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Losses
        self.loss_fn_disc = tf.keras.losses.CategoricalCrossentropy() if self.discretize else tf.keras.losses.MeanSquaredError()
        self.loss_fn_reconstruction = tf.keras.losses.MeanSquaredError()

        self.loss_disc_name = self.loss_fn_disc.get_config()['name']
        self.loss_reconstruction_name = self.loss_fn_reconstruction.get_config()['name']

        # Optimizers
        self.optimizer_disc = tf.keras.optimizers.Adam(learning_rate=self.lr_disc)
        self.optimizer_ae = tf.keras.optimizers.Adam(learning_rate=self.lr_ae)

        # Checkpoint manager
        checkpoint_path = os.path.join(self.root_folder, 'checkpoints', 'train', self.model_id)

        self.ckpt = tf.train.Checkpoint(autoencoder=self.autoencoder,
                                **self.disc_models_dict,
                                optimizer_ae=self.optimizer_ae,
                                optimizer_disc=self.optimizer_disc,
                                epoch=tf.Variable(0))

        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt, directory=checkpoint_path, max_to_keep=3)

        # Lambda values
        if (len(self.lambda_values) == 1) and self.multi_disc:
            self.lambda_values[0] = self.lambda_values[0]/len(self.cond_labels)
            self.lambda_values *= len(self.cond_labels)

        self.lambda_dict = dict(zip(self.cond_labels, self.lambda_values))

        # Dynamic learning
        self.dynamic_learning = (self.dl_no_change is not None) and (self.dl_lambda_steps is not None)

        # Train from GANDALF original code
        train_utils.train(
            dataset, self.model_id, self.epochs,
            search_ckpt, ckpt_step, saved_model_path,
            self.dynamic_learning, self.dl_no_change, self.dl_lambda_steps,
            verbose, extra_params=self)

    def eval_training(self, dataset_test, verbose=1):
        # Evaluate from GANDALF original code
        ae_loss, reconstruction_loss, disc_loss = train_utils.eval_training(dataset_test, verbose, extra_params=self)

        return ae_loss, reconstruction_loss, disc_loss

    def update_results_csv(self, reconstruction_loss, disc_loss, ae_loss):

        dataset_dir = self._dataloader.get_dataset_dir()
        dataset_name = self._dataloader.get_dataset_name()
        dataset_rows = self._X_train_shape[0] + self._X_test_shape[0]
        training_shape = '{:}x{:}'.format(self._X_train_shape[0], self._X_train_shape[1])
        test_shape = '{:}x{:}'.format(self._X_test_shape[0], self._X_test_shape[1])

        X_normalization = 'Quantile Transformer' if self.normalize_spectra else None
        params_normalization = 'Quantile Transformer'
        model_type = 'Adversarial CVAE'
        model_description = ''
        opt_name_ae = self.optimizer_ae.get_config()['name']
        # lr_ae already defined
        opt_name_disc = self.optimizer_disc.get_config()['name']
        # lr_disc already defined
        encoder_hidden_layers = self.encoder_hidden_layer_sizes
        decoder_hidden_layers = self.decoder_hidden_layer_sizes

        ae_arquitecture = '{:}+{:}-{:}-{:}-{:}+{:}-{:}-{:}-{:}'.format(self._data_dim, len(self.cond_labels),
                                                                    encoder_hidden_layers[0], encoder_hidden_layers[1],
                                                                    self.latent_size, len(self.cond_labels),
                                                                    decoder_hidden_layers[0], decoder_hidden_layers[1],
                                                                    self._data_dim)
        disc_hidden_layers = self.disc_hidden_layer_sizes
        disc_arquitecture = '{:}-{:}-{:}-{:}'.format(self.latent_size,
                                                    disc_hidden_layers[0], disc_hidden_layers[1],
                                                    self._output_discriminator)

        generate_new_result_row(file_path='results/results.csv', model_id=self.model_id, root_folder=self.root_folder, data_loader=self.data_loader, dataset_dir=dataset_dir,
                            dataset_name=dataset_name, dataset_rows=dataset_rows, training_shape=training_shape,
                            test_shape=test_shape, shuffle=self.shuffle, discretize=self.discretize, nbins=self.nbins,
                            batch_norm=self.batch_norm, conv_disc=self.conv_disc, multi_disc=self.multi_disc,
                            X_normalization=X_normalization, params_normalization=params_normalization,
                            model_type=model_type, model_description=model_description,
                            reconstruction_loss_name=self.loss_reconstruction_name, discriminator_loss_name=self.loss_disc_name,
                            opt_name_ae=opt_name_ae, lr_ae=self.lr_ae, opt_name_disc=opt_name_disc, lr_disc=self.lr_disc, seed=self.seed,
                            encoder_hidden_layers=encoder_hidden_layers, decoder_hidden_layers=decoder_hidden_layers,
                            input_without_params=self.input_without_params, latent_size=self.latent_size, ae_arquitecture=ae_arquitecture,
                            disc_hidden_layers=disc_hidden_layers, disc_arquitecture=disc_arquitecture,
                            labels=str(self.labels).replace('[','').replace(']','').replace("'",'').replace(', ','-'),
                            cond_labels=str(self.cond_labels).replace('[','').replace(']','').replace("'",'').replace(', ','-'),
                            lambda_values=self.lambda_values, dynamic_learning=self.dynamic_learning, dl_no_change=self.dl_no_change,
                            dl_lambda_steps=self.dl_lambda_steps, epochs=self.ckpt.epoch.numpy(), batch_size=self.batch_size,
                            reconstruction_loss=reconstruction_loss, discriminator_loss=disc_loss, ae_loss=ae_loss,
                            verbose=1)


    def save_visualization_data(self, dataset_train, dataset_test, update_json=False):

        # The data is extracted from the datasets
        X_train = np.concatenate([x for x, *_ in dataset_train], axis=0)
        X_test = np.concatenate([x for x, *_ in dataset_test], axis=0)
        cond_params_train = np.concatenate([y for _, y, *_ in dataset_train], axis=0)
        cond_params_test = np.concatenate([y for _, y, *_ in dataset_test], axis=0)
        if self.input_without_params:
            z_train = np.concatenate([self.autoencoder.get_layer('encoder')(x, training=False).numpy() for x, y, *_ in dataset_train])
            z_test = np.concatenate([self.autoencoder.get_layer('encoder')(x, training=False).numpy() for x, y, *_ in dataset_test])
        else:
            z_train = np.concatenate([self.autoencoder.get_layer('encoder')([x, y], training=False).numpy() for x, y, *_ in dataset_train])
            z_test = np.concatenate([self.autoencoder.get_layer('encoder')([x, y], training=False).numpy() for x, y, *_ in dataset_test])
        decoded_train = np.concatenate([self.autoencoder([x, y], training=False).numpy() for x, y, *_ in dataset_train])
        decoded_test = np.concatenate([self.autoencoder([x, y], training=False).numpy() for x, y, *_ in dataset_test])

        # ## Save results for visual-autoencoder
        base_folder_path = 'results'
        data_folder = 'data'

        data_folder_path = os.path.normpath(os.path.join(self.root_folder, base_folder_path, data_folder, self.model_id))

        # Create folder if doesn't exist
        if not os.path.exists(data_folder_path):
            print('Saving data. Folder doesn\'t exist! Creating in \'{:}\''.format(os.path.abspath(data_folder_path)))
            os.makedirs(data_folder_path)
        else:
            print('Saving data in {:}'.format(os.path.abspath(data_folder_path)))


        y_labels = self._dataloader.get_name_labels()

        # Saving data
        np.save(os.path.join(data_folder_path, 'X_train'), X_train)
        np.save(os.path.join(data_folder_path, 'X_test'), X_test)
        np.save(os.path.join(data_folder_path, 'cond_params_train'), cond_params_train)
        np.save(os.path.join(data_folder_path, 'cond_params_test'), cond_params_test)
        np.save(os.path.join(data_folder_path, 'params_train'), self._dataloader.get_original_params_train())
        np.save(os.path.join(data_folder_path, 'params_test'), self._dataloader.get_original_params_test())
        np.save(os.path.join(data_folder_path, 'z_train'), z_train)
        np.save(os.path.join(data_folder_path, 'z_test'), z_test)
        np.save(os.path.join(data_folder_path, 'decoded_train'), decoded_train)
        np.save(os.path.join(data_folder_path, 'decoded_test'), decoded_test)
        np.save(os.path.join(data_folder_path, 'ids_train'), self._dataloader.get_ids_train())
        np.save(os.path.join(data_folder_path, 'ids_test'), self._dataloader.get_ids_test())
        np.save(os.path.join(data_folder_path, 'axis_labels'), self._dataloader.get_axis_labels())
        np.save(os.path.join(data_folder_path, 'params_names'), y_labels)
        np.save(os.path.join(data_folder_path, 'cond_params_names'), self.cond_labels)

        # Saving scalers
        dump(self._dataloader.get_scaler_X() if self.normalize_spectra else None, open(os.path.join(data_folder_path, 'X_scaler.pkl'), 'wb'))
        dump(self._dataloader.get_scaler_params(), open(os.path.join(data_folder_path, 'param_scaler.pkl'), 'wb'))

        # Update json
        if update_json:
            _json_path = os.path.join(self.root_folder, '..', 'visualization', 'configuration.json')
            with open(_json_path, 'r') as jsonfile:
                _config_data = json.load(jsonfile)

            _config_data['model']['model_id'] = self.model_id

            with open(_json_path, 'w') as jsonfile:
                json.dump(_config_data, jsonfile, indent=4)

    def show_arguments(self):
        print("----------------- Arguments -----------------")
        for key, value in self._config.items():
            if type(value) is dict:
                print("---------------------------------------------")
                print("Configurations for {}:".format(str(key)))
                for k, v in value.items():
                    print("{:>30}: {:<30}".format(str(k), str(v)))
            else:
                print("{}: {}".format(str(key), str(value)))
        print("---------------------------------------------")
