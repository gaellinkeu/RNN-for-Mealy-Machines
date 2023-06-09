import os

for i in range(3):
    id = i
    os.system(f'python datagenerator.py --id={id}')
    os.system(f'python train_rnn.py --id={id}')
    os.system(f'python extract_mealy --id={id}')