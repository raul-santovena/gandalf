import pandas as pd
import argparse
import shutil
import os

CHECKPOINT_PATH = 'gandalf/train/checkpoints/train/'
LOGS_PATH = 'gandalf/train/logs/fit/'
RESULTS_DATA_PATH = 'gandalf/train/results/data/'
MODELS_PATH = 'gandalf/train/results/models/'
TEST_TSNE_PATH = 'gandalf/train/tests/tsne/'
PATHS = [CHECKPOINT_PATH, LOGS_PATH, RESULTS_DATA_PATH, MODELS_PATH, TEST_TSNE_PATH]

RESULTS_CSV = 'gandalf/train/results/results.csv'

def delete_experiment(id):
    print("-----------------------------------")
    print("Deleting experiment with id: ", id)

    dirs = [path + id for path in PATHS]

    for directory in dirs:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
        else:
            print(f"Deleting {directory}...")
            shutil.rmtree(directory)

    print("Deleting row from results.csv...\n")
    results = pd.read_csv(RESULTS_CSV)
    results = results[results.model_id != id]
    results.to_csv(RESULTS_CSV, index=False)

def cli():
    parser = argparse.ArgumentParser(prog='delete_cli', description='Delete files attached to a experiment id')

    parser.add_argument('ids', metavar='id', type=str, nargs='+',
                    help='experiment ids to delete')
    
    args = parser.parse_args()
    ids = args.ids

    print("Deleting experiments: ", ids)
    print("")

    for id in ids:
        delete_experiment(id)

    print("Done!")

if __name__ == "__main__":
    cli()