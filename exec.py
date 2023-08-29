import os
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_machines", type=int, default=10)
    parser.add_argument("--times", type=int, default=1)
    return parser.parse_args()

def threshold_study():
    args = parse_args()

    sim_thresholds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

    n_machines = args.n_machines
    for j in range(args.times):
        print(f'\n\n The times: {j+1}\n\n')
        for i in range(n_machines):
            id = i
            times = j
            os.system(f'python fsm_initialization.py --id={id} --n_states={id+2}')
            os.system(f'python train_rnn.py --id={id}')
            for sim_threshold in sim_thresholds:
                os.system(f'python extract_mealy2.py --id={id} --sim_threshold={sim_threshold}')

if __name__ == "__main__":
    args = parse_args()

    n_machines = args.n_machines
    for i in range(n_machines):
        os.system(f'python fsm_initialization.py --id={i} --n_states={i+2}')
   
    for i in range(n_machines):
        os.system(f'python train_rnn.py --id={i}')
    
    for i in range(n_machines):
        os.system(f'python extract_mealy.py --id={i}')