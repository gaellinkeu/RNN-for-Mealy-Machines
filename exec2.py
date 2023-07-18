import os
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_machines", type=int, default=10)
    parser.add_argument("--times", type=int, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    sim_thresholds = [0.2, 0.4, 0.6, 0.8, 0.9]
    n_machines = args.n_machines
    for j in range(2,10):
        for i in sim_thresholds:
            print(f'\n\n\n\n The machine {j} and the threshold {i}\n\n\n')
            os.system(f'python extract_mealy.py --id={j} --eval=\'labels\' --sim_threshold={i}')