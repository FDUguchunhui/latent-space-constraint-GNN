import argparse
import subprocess

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Define the different split_ratio and dataset choices
    # split from 0.1 to 0.9 by 0.1
    datasets = ['Cora']

    # Iterate over the choices
    for dataset in datasets:
        for i in range(10):  # Run three times
            seed = args.seed + i  # Increment seed by 1 each time
            # Run main.py with the current split_ratio, dataset, and seed
            subprocess.run(['python', 'main_vgae.py', '--dataset', dataset, '--seed', str(seed)])
            # sleep for 1 seconds
            subprocess.run(['sleep', '1'])