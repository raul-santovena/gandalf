import pandas as pd
import argparse
import shutil
import os

CHECKPOINT_PATH = 'checkpoints/train/'
LOGS_PATH = 'logs/fit/'
RESULTS_DATA_PATH = 'results/data/'
MODELS_PATH = 'results/models/'
TEST_TSNE_PATH = 'tests/tsne/'
PATHS = [CHECKPOINT_PATH, LOGS_PATH, RESULTS_DATA_PATH, MODELS_PATH, TEST_TSNE_PATH]

RESULTS_CSV_PATH = 'results/results.csv'

def delete_experiment(id, root_folder):
    print("-----------------------------------")
    print("Deleting experiment with id: ", id)

    dirs = [os.path.normpath(os.path.join(root_folder, path, id)) for path in PATHS]

    for directory in dirs:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
        else:
            print(f"Deleting {directory}...")
            shutil.rmtree(directory)

    print("Deleting row from results.csv...\n")
    results_csv_full_path = os.path.join(root_folder, RESULTS_CSV_PATH)
    results = pd.read_csv(results_csv_full_path)
    results = results[results.model_id != id]
    results.to_csv(results_csv_full_path, index=False)

def cli():
    parser = argparse.ArgumentParser(prog='delete_cli', description='Delete files attached to a experiment id')

    parser.add_argument('ids', metavar='id', type=str, nargs='+',
                    help='experiment ids to delete')
    parser.add_argument('--root_folder', type=str, default=os.path.dirname(__file__),
                        help='root folder where to search for experiments')
    
    args = parser.parse_args()
    ids = args.ids
    root_folder = args.root_folder

    print("Deleting experiments: ", ids)
    print("")

    for id in ids:
        delete_experiment(id, os.path.normpath(root_folder))

    print("Done!")

if __name__ == "__main__":
    cli()