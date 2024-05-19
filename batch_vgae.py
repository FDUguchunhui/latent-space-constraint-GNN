import argparse
import subprocess

if __name__ == '__main__':

    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='batch run')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    datasets = ['Cora', 'CiteSeer']
    # build grid list using split_ratio and dataset

    # Iterate over the choices
    for dataset in datasets:
        for i in range(3):  # Run three times
            seed = args.seed + i  # Increment seed by 1 each time
            # Run main.py with the current split_ratio, dataset, and seed
            subprocess.run(['python', 'main_vgae.py', '--dataset', dataset, '--seed', str(seed)])
            # sleep for 1 seconds
            subprocess.run(['sleep', '1'])