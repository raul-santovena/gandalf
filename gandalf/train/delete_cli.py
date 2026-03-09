import argparse
import sys
import os

# My modules
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.normpath('../../'))) # To support the use of the tool without packaging
from gandalf.utils.delete_experiment import delete_experiment

def cli():
    parser = argparse.ArgumentParser(prog='delete_cli', description='Delete files attached to a experiment id')

    parser.add_argument('ids', metavar='id', type=str, nargs='+',
                    help='experiment ids to delete')
    parser.add_argument('--root_folder', type=str, default=os.path.dirname(__file__),
                        help='root folder where to search for experiments (default: current directory)')
    parser.add_argument('--force', action='store_true',
                        help='force deletion even if experiment_id is not in results.csv')

    args = parser.parse_args()
    ids = args.ids
    root_folder = args.root_folder
    force = args.force

    print(f"Target Root Folder: {root_folder}")
    print(f"Experiment IDs to Delete: {', '.join(ids)}")
    print(f"Force Mode: {'Enabled' if args.force else 'Disabled'}")
    print("===================================\n")

    if not os.path.isdir(root_folder):
        print(f"Error: The specified root folder does not exist or is not a directory: '{root_folder}'")
        sys.exit(1)

    for id in ids:
        delete_experiment(id, os.path.normpath(root_folder), force=force)

    print("All specified experiment IDs have been processed.")

if __name__ == "__main__":
    cli()