import os
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_machines", type=int, default=2)
    parser.add_argument("--times", type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    n_machines = args.n_machines
    for j in range(args.times):
        print(f'\n\n\n\n The times: {j}\n\n\n')
        for i in range(n_machines):
            id = i
            times = j
            os.system(f'python fsm_initialization.py --id={id} --n_states={id+2} --times={times+1}')
            os.system(f'python train_rnn.py --id={id} --times={times+1}')
            if i == 0:
                os.system(f'python extract_mealy.py --new_runtime=1 --id={id} --times={times+1}')
            else:
                os.system(f'python extract_mealy.py --id={id} --times={times+1}')