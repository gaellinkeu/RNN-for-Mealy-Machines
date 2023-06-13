import os
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_machines", type=int, default=7)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    n_machines = args.n_machines

    for i in range(n_machines):
        id = i
        os.system(f'python datagenerator.py --id={id} --n_states={id+2}')
        os.system(f'python train_rnn.py --id={id}')
        if i == 0:
            os.system(f'python extract_mealy.py --new_runtime=1 --id={id}')
        else:
            os.system(f'python extract_mealy.py --id={id}')