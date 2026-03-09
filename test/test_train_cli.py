from constants import RESULTS_PATH
import subprocess
import shutil
import os

class TestTrainCLI:

    def teardown_method(self, method):
        """Teardown method to clean up after each test."""
        if os.path.exists(RESULTS_PATH):
            shutil.rmtree(RESULTS_PATH)

    def test_train_cli(self):
        COMMAND = f"python gandalf/train/train_cli.py --root_folder {RESULTS_PATH} --epochs 5 --survey_data_path gandalf/train/data/sample --params teff logg --cond_params teff"

        result = subprocess.run(COMMAND, shell=True, capture_output=True, text=True)

        assert result.returncode == 0, f"Command failed with error: {result.stderr}"

        assert os.path.exists(RESULTS_PATH)
        assert os.path.exists(os.path.join(RESULTS_PATH, 'checkpoints'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'logs'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'results'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'results', 'results.csv'))
        assert os.path.exists(os.path.join(RESULTS_PATH, 'results', 'data'))
