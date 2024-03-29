import os
from scoring import *
from utils import fsm_equivalence

def save_results(filepath, parameters):

    f1 = open(filepath, 'a+')
    lines = f1.readlines()
    if os.path.getsize(filepath) == 0:
        f1.write('ID,Time,data,sim_threshold,Final_acc,Final_dev_acc,initial_train_acc,initial_dev_acc,all_merges,correct_merges,state_set_size,equivalence\n')

    f1.write(f'{parameters['id']},{parameters['seed']},')
    f1.write(f'{parameters['n']},{parameters['sim_threshold']},')
    # Evaluate performance
    _acc = score_all_prefixes(parameters['merged_fsm'], parameters['inputs'], parameters['outputs'])
    parameters['train_acc'][parameters['seed']].append(_acc)
    f1.write(f'{_acc},')
    _dev_acc = score_whole_words(parameters['merged_fsm'], parameters['dev_inputs'], parameters['dev_outputs'])
    parameters['dev_acc'][parameters['seed']].append(_dev_acc)
    f1.write(f'{_dev_acc},')

    _acc = score_all_prefixes(parameters['init_fsm'], parameters['inputs'], parameters['outputs'])
    parameters['init_train_acc'][parameters['seed']].append(_acc)
    f1.write(f'{_acc},')
    parameters['_init_dev_acc'] = score_whole_words(parameters['init_fsm'], parameters['dev_inputs'], parameters['dev_outputs']) 
    parameters['init_train_acc'][parameters['seed']].append(parameters['init_train_acc'])
    f1.write(f'{parameters['_init_dev_acc']},')

    f1.write(f'{parameters['all_merges']},{parameters['correct_merges']},')

    parameters['state_set_size'][parameters['seed']].append(len(parameters['merged_fsm.states']))
    f1.write(f'{len(parameters['merged_fsm.states'])},')

    equivalence = fsm_equivalence(parameters['expected_fsm'], parameters['merged_fsm'])
    f1.write(f'{equivalence}\n')