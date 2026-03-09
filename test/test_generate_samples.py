from constants import CONFIG_FILE_PATH, RESULTS_PATH, ID, PARAMS_PATH, X_PATH, LATENT_DIM, FEATURES_LEN
import numpy as np
import shutil
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from gandalf.train_adapter.gandalf_train import GandalfTrain
from gandalf.utils import generate_samples, generate_z, generate_samples_from_z

# Constants for the test
INPUT_WITHOUT_PARAMS = False
DATA_LIMIT = 4

@pytest.fixture(scope="class")
def setup_train(request):
    """
    This fixture prepares the environment, trains the model,
    and provides the necessary objects to the class tests.
    Cleanup is performed automatically at the end.
    """
    # 1. Setup
    gandalf_train = GandalfTrain.from_config_file(CONFIG_FILE_PATH)
    dataset_train, dataset_test = gandalf_train.create_datasets()
    gandalf_train.create_autoencoder()
    gandalf_train.create_discriminator()
    gandalf_train.train(dataset_train)
    ae_loss, reconstruction_loss, disc_loss = gandalf_train.eval_training(dataset_test)
    gandalf_train.update_results_csv(ae_loss, reconstruction_loss, disc_loss)
    gandalf_train.save_visualization_data(dataset_train, dataset_test)

    params = np.load(PARAMS_PATH)[:DATA_LIMIT]
    data = np.load(X_PATH)[:DATA_LIMIT]

    yield params, data

    # 2. Cleanup
    if os.path.exists(RESULTS_PATH):
        shutil.rmtree(RESULTS_PATH)

class TestGenerateSamples():

    def test_generate_z_and_samples_from_z(self, setup_train):
        params, data = setup_train

        # : is to get all the samples, 0 is to get the first parameter (teff)
        cond_params = params.copy()[:,0]
        cond_params = np.expand_dims(cond_params, axis=1)

        z = generate_z(RESULTS_PATH + "results", ID, data, cond_params, input_without_params=INPUT_WITHOUT_PARAMS)

        assert z.shape == (DATA_LIMIT, LATENT_DIM)

        # Reconstruct samples from the latent space using new parameters
        target_params = cond_params.copy()
        target_params += 800 # increase the temperature by 800 K

        reconstructed = generate_samples_from_z(RESULTS_PATH + "results", ID, z, target_params)

        assert reconstructed.shape == (DATA_LIMIT, FEATURES_LEN)

    def test_generate_samples(self, setup_train):
        params, data = setup_train

        cond_params = params.copy()[:,0]
        cond_params = np.expand_dims(cond_params, axis=1)
        target_params = cond_params.copy()
        target_params += 800

        reconstructed = generate_samples(RESULTS_PATH + "results", ID, data, cond_params, target_params, input_without_params=INPUT_WITHOUT_PARAMS)

        assert reconstructed.shape == (DATA_LIMIT, FEATURES_LEN)