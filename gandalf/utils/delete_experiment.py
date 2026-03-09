import pandas as pd
import shutil
import os

# Default paths for storing experiment-related files
CHECKPOINT_PATH = 'checkpoints/train/'
LOGS_PATH = 'logs/fit/'
RESULTS_DATA_PATH = 'results/data/'
MODELS_PATH = 'results/models/'
TEST_TSNE_PATH = 'tests/tsne/'
PATHS = [CHECKPOINT_PATH, LOGS_PATH, RESULTS_DATA_PATH, MODELS_PATH, TEST_TSNE_PATH]

RESULTS_CSV_PATH = 'results/results.csv'

def delete_experiment(experiment_id, root_folder, force=False):
    """
    Deletes experiment-related files and its entry from a results CSV.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment to be deleted.
    root_folder : str
        Root directory where the experiment files are stored.
    force : bool, optional
        If True, forces deletion even if the experiment_id is not found in the results CSV.
        Default is False, which will raise an error if the experiment_id does not exist.
    """
    print("-----------------------------------")
    print(f"Processing deletion for experiment id: {experiment_id}")

    dirs = [os.path.normpath(os.path.join(root_folder, path, experiment_id)) for path in PATHS]
    results_csv_full_path = os.path.join(root_folder, RESULTS_CSV_PATH)
    results_df = pd.read_csv(results_csv_full_path)

    if not force and str(experiment_id) not in results_df['model_id'].astype(str).values:
        print(f"Error: experiment_id '{experiment_id}' does not exist in {RESULTS_CSV_PATH}.")
        return

    deleted_any_dirs = False
    for directory_path in dirs:
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist. Skipping.")
        else:
            print(f"Deleting {directory_path}...")
            shutil.rmtree(directory_path)
            deleted_any_dirs = True

    if not deleted_any_dirs:
        print("No directories found or deleted for this experiment.")

    print("Deleting row from results.csv...")
    results_df = results_df[results_df.model_id != experiment_id]
    results_df.to_csv(results_csv_full_path, index=False)

    print(f"\nFinished processing experiment {experiment_id}.\n")