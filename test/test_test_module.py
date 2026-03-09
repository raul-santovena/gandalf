from constants import CONFIG_FILE_PATH, COND_LABELS, RESULTS_PATH
import shutil
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from gandalf.train_adapter.gandalf_train import GandalfTrain
from gandalf.test import tsne_comparison, cond_prediction_comparison, no_cond_prediction_comparison

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

    yield gandalf_train

    # 2. Cleanup
    if os.path.exists(RESULTS_PATH):
        shutil.rmtree(RESULTS_PATH)

class TestTestModule():

    def test_tsne_comparison(self, setup_train):
        ROOT_FOLDER="test/test-results/"
        ID = "test-experiment"
        RANDOM_STATE = 42
        EXPORT_GRAPH = True

        tsne_comparison(id_=ID, root_folder=ROOT_FOLDER, name="TEST-TSNE", cache='tests/tsne', random_state=RANDOM_STATE,
                                export_graph=EXPORT_GRAPH, backend="matplotlib")

        assert os.path.exists(os.path.join(ROOT_FOLDER, 'tests', 'tsne', ID, 'x_{}.npy'.format(RANDOM_STATE)))
        assert os.path.exists(os.path.join(ROOT_FOLDER, 'tests', 'tsne', ID, 'z_{}.npy'.format(RANDOM_STATE)))
        assert os.path.exists(os.path.join(ROOT_FOLDER, 'tsne_comparison_{}_{}.png').format(ID, COND_LABELS[0]))

    def test_cond_prediction_comparison(self, setup_train):
        ROOT_FOLDER="test/test-results/"
        ID = "test-experiment"
        RANDOM_STATE = 42

        X_HIDDEN_LAYER_SIZES = [200, 100]
        Z_HIDDEN_LAYER_SIZES = [200, 100]
        REPETITIONS = 1
        MAX_ITER = 600
        LATEX_FORMAT = False

        cond_prediction_comparison(model_ids=[ID], root_folder=ROOT_FOLDER,
                                       x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                       repetitions=REPETITIONS, iterations_mlp=MAX_ITER,
                                       latex_format=LATEX_FORMAT, random_state=RANDOM_STATE)

        no_cond_prediction_comparison(model_ids=[ID], root_folder=ROOT_FOLDER,
                                             x_hidden_layers=X_HIDDEN_LAYER_SIZES, z_hidden_layers=Z_HIDDEN_LAYER_SIZES,
                                             repetitions=REPETITIONS, iterations_mlp=MAX_ITER,
                                             latex_format=LATEX_FORMAT, random_state=RANDOM_STATE)
