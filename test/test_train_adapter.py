from constants import CONFIG_FILE_PATH, TRAIN_DATASET_LENGTH, TEST_DATASET_LENGTH, FEATURES_LEN, COND_LABELS, LATENT_DIM, RESULTS_PATH
import tensorflow as tf
import numpy as np
import shutil
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from gandalf.train_adapter.gandalf_train import GandalfTrain

class TestTrainAdapter:

    def teardown_method(self, method):
        """Teardown method to clean up after each test."""
        if os.path.exists(RESULTS_PATH):
            shutil.rmtree(RESULTS_PATH)

    def test_initialization(self):
        gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)
        assert gandalf_train is not None

    def test_create_datsets(self):
        gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)

        # Set batch size to 1 to match TRAIN_DATASET_LENGTH and TEST_DATASET_LENGTH
        gandalf_train.batch_size = 1

        dataset_train, dataset_test = gandalf_train.create_datasets()

        assert isinstance(dataset_train, tf.data.Dataset)
        assert isinstance(dataset_test, tf.data.Dataset)

        assert dataset_train.cardinality().numpy() == TRAIN_DATASET_LENGTH
        assert dataset_test.cardinality().numpy() == TEST_DATASET_LENGTH

        train_data_dim = dataset_train.element_spec[0].shape[-1]
        test_data_dim = dataset_test.element_spec[0].shape[-1]

        assert train_data_dim == FEATURES_LEN
        assert test_data_dim == FEATURES_LEN

    def test_create_autoencoder(self):
        gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)

        # previous steps are required
        gandalf_train.create_datasets()

        gandalf_train.create_autoencoder()
        autoencoder = gandalf_train.autoencoder

        assert isinstance(autoencoder, tf.keras.Model)

        features_len = autoencoder.input_shape[0][-1]
        cond_labels = autoencoder.input_shape[1][-1]

        assert features_len == FEATURES_LEN
        assert cond_labels == len(COND_LABELS)

    def test_create_discriminator(self):
        gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)

        # previous steps are required
        gandalf_train.create_datasets()
        gandalf_train.create_autoencoder()

        gandalf_train.create_discriminator()

        # As multi_disc: True it contains one discriminator per conditional label
        disc_models_dict = gandalf_train.disc_models_dict

        for cond_label in COND_LABELS:
            disc_name = f"{cond_label}_discriminator"

            assert disc_name in disc_models_dict.keys()
            assert isinstance(disc_models_dict[disc_name], tf.keras.Model)
            assert disc_models_dict[disc_name].input_shape[-1] == LATENT_DIM

    def test_train_and_eval(self):
        """
        Test the full training and evaluation process.

        The rest of the code will be tested in this single function,
        as the train function cannot be called more than once in the same execution.
        This function uses the @tf.function decorator, which will cause the following error to be thrown:
        ValueError: tf.function only supports singleton tf.Variables created on the first call"""

        gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)

        # previous steps are required
        dataset_train, dataset_test = gandalf_train.create_datasets()
        gandalf_train.create_autoencoder()
        gandalf_train.create_discriminator()

        # Train the model
        gandalf_train.train(dataset_train)

        assert os.path.exists(RESULTS_PATH)
        assert os.path.exists(os.path.join(RESULTS_PATH, 'checkpoints'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'logs'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'results'))

        # Evaluate training
        ae_loss, reconstruction_loss, disc_loss = gandalf_train.eval_training(dataset_test)

        assert isinstance(ae_loss.numpy(), np.float32)
        assert isinstance(reconstruction_loss.numpy(), np.float32)
        assert isinstance(disc_loss.numpy(), np.float32)

        # Save data and test update_results_csv and save_visualization_data
        gandalf_train.update_results_csv(ae_loss, reconstruction_loss, disc_loss)

        assert os.path.exists(os.path.join(RESULTS_PATH, 'results', 'results.csv'))

        gandalf_train.save_visualization_data(dataset_train, dataset_test)
        assert os.path.exists(os.path.join(RESULTS_PATH, 'results', 'data'))
