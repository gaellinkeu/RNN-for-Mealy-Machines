import argparse
from datetime import date
import sys
from copy import deepcopy
import pickle
import os
import numpy as np
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt

from mealy_trie import Trie
from mealy_machine import Mealy
from utils import *
from model import load_model, get_rnn_states_and_outputs
from create_plot import create_plot
from loading_data import load_data
from scoring import *
from merging import *
from training_data_preprocessing import preprocessing, one_hot_decoding
from save_states import save_states
from save_results import save_extraction_results, save_final_report
#from IPython.display import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=2)
    parser.add_argument("--dev_length", type=int, default=1000)
    parser.add_argument("--n_train_low", type=int, default=100)
    parser.add_argument("--n_train_high", type=int, default=101)
    parser.add_argument("--word_dev_low", type=int, default=1)
    parser.add_argument("--word_dev_high", type=int, default=100)
    parser.add_argument("--sim_threshold", type=float, default=.9)
    parser.add_argument("--find_threshold", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--similarity_effect", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--hidden_size", type=float, default=10)
    parser.add_argument('--eval', type=str, default="preds")
    parser.add_argument('--epoch', type=str, default="best")    
    parser.add_argument('--times', type=int, default=0)
    parser.add_argument("--new_runtime", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__" :
    args = parse_args()
    id = args.id
    init_train_acc, init_dev_acc, train_acc, dev_acc = {}, {}, {}, {}
    state_set_size = {}

    print('\n\n\n'+'*'*20+f' ID {id} : '+' EXTRACTION OF MEALY MACHINE FROM RNN '+'*'*20+'\n\n\n')

    
    fsm_filepath = f'./FSMs/fsm{id}.txt'
    expected_fsm = getFsm(fsm_filepath)

    data_filepath = f'./datasets/dataset{id}.txt'
            
        
    if args.similarity_effect:
        n_train = range(100,101)
    else:
        n_train = range(args.n_train_low, args.n_train_high)
    
    for seed in range(args.seeds):
        random.seed(seed)
        dev_inputs = []
        dev_outputs = []
        for _ in range(args.dev_length):
            word = randomWord(args.word_dev_low, args.word_dev_high, expected_fsm.inputAlphabet)
            dev_inputs.append(word)
            dev_outputs.append(expected_fsm.return_output(word))

        max_length_dev = len(max(dev_inputs, key=len))

        sim_threshold = args.sim_threshold

        # result filepath
        os.makedirs(f"./Results/{id}",exist_ok=True)
        results_filepath = f'./Results/{id}/extraction.txt'
    
        # Initialize variables where the results will be saved
        init_train_acc[seed], init_dev_acc[seed], train_acc[seed], dev_acc[seed] = [], [], [], []
        state_set_size[seed] = []

        for n in n_train:
            print(f'The length of the constructing dataset: {n} inputs')

            """dev_inputs = inputs[split_index:]
            dev_outputs = outputs[split_index:]
            max_length_dev = len(max(dev_inputs, key=len))"""
            inputs, outputs, max_length_train = load_data(data_filepath)

            # Data Spliting
            val_inputs = inputs[n:2*n]
            val_outputs = outputs[n:2*n]
            train_inputs = inputs[:n]
            train_outputs = outputs[:n]

            # train_inputs = n_uplets(['a','b'], 4)
            # train_outputs = n_uplets(['0','1'], 4)

            max_length_train_inputs = len(max(train_inputs, key=len))

            # Data preprocessing
            inputs_, outputs_, n_tokens, n_labels, mask = preprocessing(train_inputs, train_outputs)
            print("\n\--> Data Preprocessing... Done\n")


            # Retrieve the trained model 
            trained_model = load_model(args.id, n_tokens, max_length_train_inputs, 10, args.hidden_size, n_labels)
            print('\--> Model retrieving... Done\n')

            # Predicting the outputs of dataset input sequences
            rnn_states, pred_outputs = get_rnn_states_and_outputs(trained_model, inputs_, mask)
            
            # Evaluate the trained RNN on the development set
            test_acc = trained_model.evaluate(inputs_, outputs_, verbose=0)
            print("\n The unmerged tree testing accuracy: %.2f%%" % (test_acc[1]*100))

            # Build the trie
            # if eval eval is preds, used outputs predicted from the RNN
            # else, use outputs in the dataset
            if args.eval == 'preds' :
                redundant_fsm = build_trie_from_dict(id, train_inputs, pred_outputs)
            else:
                redundant_fsm = build_trie_from_dict(id, train_inputs, train_outputs)
            print('\--> Trie Building... Done\n')


            print('\--> Checking if the trie get the right ouput for each input... Done\n')

            
            print('\--> Getting states... Done\n')

            # Initialize variables for states mapping
            idx = [redundant_fsm.return_states(sent) for sent in train_inputs] # maps strings to states
            n_states = len(redundant_fsm.states)
            states = np.zeros((n_states, args.hidden_size))
            states_mask = np.zeros(n_states)
            print(f' The total amount of states got from the RNN : {n_states}\n')

            print('\--> States Mapping preparation... Done\n')
            
            outputs_ = [one_hot_decoding(x) for x in outputs_]
            outputs_ = np.array(outputs_)

            # Map RNN states to Tree nodes according to nodes traces when processing each word
            for i, _r in enumerate(rnn_states):
                states[idx[i]] = _r[mask[i]]
                states_mask[idx[i]] = outputs_[i][mask[i]]
            
            save_states(id, train_inputs, pred_outputs, states, idx)

            
            print('\--> States Mapping... Done\n')

            init_fsm = deepcopy(redundant_fsm)

            print('\--> Merging Preparation... Done\n')

            if(args.find_threshold):
                # Find the optimat similarity threshold
                print("We are finding the optimal threshold")
                merged_fsm, all_merges, correct_merges, sim_threshold, _ = cross_validate(.7, 1., redundant_fsm, states, states_mask, val_inputs, val_outputs)
                print(sim_threshold)
            else:
                # Use a fixed similarity threshold
                print(f'We used the threshold : {sim_threshold}')
                merged_fsm, all_merges, correct_merges = cosine_merging(redundant_fsm, states, threshold=sim_threshold)
            print('\--> Merging stage... Done\n')

            # Print the final Mealy Machine
            merged_fsm.print(print_all=True)

            
            # MINIZATION STEP
            # merged_fsm.save(f"./FSMs_extracted_first", sim_threshold)
            # print('\--> Minimization stage... Done\n')
            # if merged_fsm.is_output_deterministic():
            #     MM_extracted_filepath = f'./FSMs_visuals/fsm{id}_{args.sim_threshold}_first_extracted.dot'
            #     f = open(MM_extracted_filepath, "w")
            #     f.write(merged_fsm.toDot())
            #     f.close()
            #     isd, st = merged_fsm.is_state_deterministic()
            #     if not isd:
            #         s = [list(x.values()) for x in st]
            #         #print(s)
            #         merged_fsm.final_merges(s)
            #     if merged_fsm.determinize():
            #         #if merged_fsm.is_complete():
            #         merged_fsm.minimize()
            # merged_fsm.save(f"./FSMs_extracted", sim_threshold)
            # merged_fsm.print(print_all=True)

            print('\--> Merged FSM saved stage... Done\n')
            

            # Evaluate performance
            _acc = score_all_prefixes(merged_fsm, inputs, outputs)
            train_acc[seed].append(_acc)

            _dev_acc = score_whole_words(merged_fsm, dev_inputs, dev_outputs)
            dev_acc[seed].append(_dev_acc)

            _init_acc = score_all_prefixes(init_fsm, inputs, outputs, 'tree')
            init_train_acc[seed].append(_init_acc)

            _init_dev_acc = score_whole_words(init_fsm, dev_inputs, dev_outputs, 'tree') 
            init_dev_acc[seed].append(_init_dev_acc)

            state_set_size[seed].append(len(merged_fsm.states))

            equivalence = fsm_equivalence(expected_fsm, merged_fsm)

            save_extraction_results(id, results_filepath, seed, n, sim_threshold, _acc, _dev_acc, _init_acc, _init_dev_acc, all_merges, correct_merges, len(merged_fsm.states), equivalence)

    # Plot the accuracy comparison between the RNN and the merged tree on both training and development sets      
    # create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_train, args.id, sim_threshold, args.epoch, args.eval)
    # print('\--> Plot saving stage... Done\n')
    


    # Check the equivalence between the expected and the obtained machine
    equivalence = fsm_equivalence(expected_fsm, merged_fsm)
    if equivalence:
        print('\n\nThe obtained FSM is *** EQUIVALENT *** to the one expected\n')
    else:
        print('\n\nThe obtained FSM is *** NOT EQUIVALENT *** to the expected FSM\n')
    
    print('\n****************  WE HAVE FINISHED   ***************')
    print(f'\n*************   THE ACCURACY IS :   {_dev_acc} %  *****************')

    # Save execution results
    os.makedirs(f"./Reports",exist_ok=True)
    report_filepath = f'./Reports/Execution{id}-{args.sim_threshold}.txt'
    save_final_report(report_filepath, id, args.times, args.sim_threshold, equivalence, all_merges, correct_merges, len(expected_fsm.states), len(merged_fsm.states), len(dev_inputs), args.word_dev_high)

    merged_fsm.save()
