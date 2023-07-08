import argparse
from datetime import date
import sys
from copy import deepcopy

from mealy_trie import Trie
from mealy_machine import Mealy
from utils import *
from mealy_machine import Mealy

import os
import numpy as np
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
from utils import *
import matplotlib.pyplot as plt
from model import Tagger, load_weights
import pickle
#from IPython.display import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_length", type=int, default=10)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=300)
    parser.add_argument("--sim_threshold", type=float, default=.95)
    parser.add_argument("--find_threshold", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--hidden_size", type=float, default=10)
    parser.add_argument('--eval', type=str, default="preds")
    parser.add_argument("--new_runtime", type=int, default=0)
    return parser.parse_args()


def score_whole_words(mealy, dataset, labels):
    acc = 0
    for word, y in zip(dataset, labels):
        acc += (mealy.return_output(word) == y)
    return (acc / len(dataset) * 100)

def score_all_prefixes(mealy, dataset, labels):
    # A bunch of code on how to determine if a label correctly corresponds
    # to the output of the mealy machine
    scores = 0
    total = 0
    for i in range(len(dataset)):
        b = f'{i+1} / {len(dataset)}'
        # \r prints a carriage return first, so `b` is printed on top of the previous line.
        #print(b + '\t' + dataset[i])
        #sys.stdout.write('\r'+b)
        #time.sleep(0.5)
        output = mealy.return_output(dataset[i])
        score = [labels[i][j] == output[j] for j in range(len(output))]
        scores += score.count(True)
        total += len(output)

    return scores/total * 100

def build_fsm_from_dict(id, dict, labels, nfa=False):
    t = Trie(dict, labels)
    my_mealy = Mealy(id, t.root.id, t.states, t.arcs)
    # states are represented in a dfs fashion
    return my_mealy

def cosine_merging(fsm, states, threshold):
     
    all_merge, correct_merge = 0, 0
    fsm_ = deepcopy(fsm)
    #print(tf.shape(sim.shape))
    #print(sim)
    #fsm_.print()
    sim1 = []
    for i in range(len(states)):
        sim1.append([])
        for j in range(i):
            sim1[i].append(cosine(states[i], states[j]))

    #blockPrint()
    similarity_bool = []
    for i in range(len(states)):
        similarity_bool.append([])
        for j in range(i):
            similarity_bool[i].append(sim1[i][j] >= threshold)

    for i in range(states.shape[0]):
        for j in range(i):
            pass_ = False
            if(i == j):
                continue

            # we double check if the two states to merge don't have different input output couple
            for x in fsm_.getInpOut(i):
                for y in fsm_.getInpOut(j):
                    if(x[0] == y[0] and x[1] != y[1]):
                        pass_ = True
            #print(f'--we have {i} and {j} and the similarity {sim[i][j]}')
            if pass_:
                continue
            
            if(sim1[i][j] >= threshold):
                #print('*****************************')
                #print(f'The states to merge {i} and {j}')
                
                x, y = fsm_.merge_states(j, i, similarity_bool)
                all_merge += x
                correct_merge += y
                #print(f'The number of states : {len(fsm_.nodes)}')
                
                #pruned += 1 - res
    #enablePrint()
    print(f'\nThe total amount of merging is: {all_merge}\n')
    print(f'\nThe total amount of correct merging is: {correct_merge}\n')
    #fsm_.print()
    fsm_.removeDuplicate()
    fsm_.id = str(fsm_.id) + 'min'
    return fsm_, all_merge, correct_merge

def cross_validate(left, right, fsm, states, states_mask, val_sents, val_gold, symmetric=False):

    max_acc = -1

    for j in np.arange(left, right, .05):
        _fsm = deepcopy(fsm)
        merged_fsm, all_merge, correct_merge = cosine_merging(_fsm, states, states_mask, j, symmetric=False)
        cur_acc = score_all_prefixes(merged_fsm, val_sents, val_gold)
        if (cur_acc > max_acc):
            max_acc = cur_acc
            opt_threshold = j
            opt_fsm = deepcopy(merged_fsm)

    return opt_fsm, all_merge, correct_merge, opt_threshold, max_acc


if __name__ == "__main__" :
    args = parse_args()
    id = args.id

    print('\n\n\n'+'*'*20+f' ID {id}: '+' EXTRACTION OF MEALY MACHINE FROM RNN '+'*'*20+'\n\n\n')

    init_train_acc, init_dev_acc, train_acc, dev_acc = {}, {}, {}, {}
    train_acc["0"] = []
    n_train = range(1)
    
    fsm_filepath = f'./FSMs/fsm{id}.txt'
    expected_fsm = getFsm(fsm_filepath)

    data_filepath = f'./datasets/dataset{id}.txt'
        
    corpus, labels = get_data(data_filepath)
    assert(len(corpus) == len(labels))

    

    """corpus = [x[:4] for x in corpus]
    labels = [x[:4] for x in labels]
    corpus, labels = corpus[:20], labels[:20]"""

    #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb', 'abb', 'aaaa', 'baaaaab', 'abababa']
    #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11', '101', '1000', '1100000', '1010101']
    
    

    print('Some words of our dataset')
    print(f'Corpus: {corpus[:5]}')
    print(f'Labels: {labels[:5]}')

    split_index = 100
    dev_corpus = corpus[split_index:]
    dev_labels = labels[split_index:]
    max_length_dev = len(max(dev_corpus, key=len))

    corpus = corpus[:split_index]
    labels = labels[:split_index]
    max_length_corpus = len(max(corpus, key=len))
    
    corpus_, labels_ = preprocessing(corpus, labels, max_length_corpus)
    dev_corpus_, dev_labels_ = preprocessing(dev_corpus, dev_labels, max_length_dev)

    dev_mask = [masking(x,'2') for x in dev_labels_]
    
    labels__ = np.array([np.array(list(x)) for x in labels_])
    mask = [masking(x) for x in corpus_]

    x_train = np.array([tokenization(x) for x in corpus_])
    train_sents = [tokenization(x) for x in corpus]
    y_train = np.array([class_mapping(x) for x in labels_])
    mask_ = np.array([masking(x) for x in corpus_])

    print("\n\--> Data Preprocessing... Done\n")

    trained_model = Tagger(4, 10, args.hidden_size, 3)
    trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filename = f'./weights/weights{id}.txt'
    with open(filename, 'rb') as f:
        weights = pickle.load(f)

    print('\--> Model definition... Done\n')

    trained_model.set_weights(weights)

    print('\--> Model Update... Done\n')

    predictions = trained_model.predict(x_train)
    predictions = predictions.argmax(axis=-1)
    pred_labels = nparray_to_string(predictions, mask)
    print(pred_labels[:5])
    print(labels[:5])
    
    print('\--> Trie Building... Done\n')

    if args.eval == 'preds' :
        redundant_fsm = build_fsm_from_dict(id, corpus, pred_labels)
        #assert(score_all_prefixes(redundant_fsm, corpus, labels) == 100.0), '\nPredictions are not the same with labels'
    else:
        redundant_fsm = build_fsm_from_dict(id, corpus, labels)
        #assert(score_all_prefixes(redundant_fsm, corpus, pred_labels) == 100.0), '\nLabels are not the same with predictions'
    #redundant_fsm.print()

 

    

    print('\--> Checking if the trie get the right ouput for each input... Done\n')

    trained_model.pop()
    trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    representations = trained_model.predict(x_train)

    print('\--> Getting states... Done\n')

    idx = [redundant_fsm.return_states(sent) for sent in corpus] # maps strings to states
    n_states = len(redundant_fsm.states)
    states = np.zeros((n_states, args.hidden_size))
    states_mask = np.zeros(n_states)

    print('\--> States Mapping preparation... Done\n')
    
    #print(idx)

    for i, _r in enumerate(representations):
        states[idx[i]] = _r[mask_[i]]
        states_mask[idx[i]] = labels__[i][mask_[i]]
    #print(states)
    #print(states[0])
    #print(states[1])
    print('\--> States Mapping... Done\n')
    init_fsm = deepcopy(redundant_fsm)

    print('\--> Merging Preparation... Done\n')

    if(args.find_threshold):
        # code for finding the good similarity threshold
        print("We are findind the optimal threshold")
        merged_fsm, all_merge, correct_merge, _, _ = cross_validate(.5, 1., redundant_fsm, states, states_mask, val_corpus, val_labels)
        
    else:
        print(f'We used the threshold : {args.sim_threshold}')
        merged_fsm, all_merge, correct_merge = cosine_merging(redundant_fsm, states, threshold=args.sim_threshold)
    print('\--> Merging stage... Done\n')
    merged_fsm.save()
    merged_fsm.print(print_all=True)
    
    # Checking the equivalence between the get
    equivalence = fsm_equivalence(expected_fsm, merged_fsm)

    if equivalence:
        print('\n\nThe obtained FSM is EQUIVALENT to the one we expected\n')
    else:
        print('\n\nThe obtained FSM is NOT EQUIVALENT to the expected FSM\n')
    
    dev_accuracy = score_whole_words(merged_fsm, dev_corpus, dev_labels)
    #_acc = 0
    print('\--> Getting the accuracy... Done\n')
    train_acc["0"].append(dev_accuracy)
    
    print('\n****************  WE HAVE FINISHED   ***************')
    print(f'\n*************   THE ACCURACY IS :   {dev_accuracy} %  *****************')

    os.makedirs(f"./Infos",exist_ok=True)
    info_filepath = f'./Infos/Execution {id}.txt'
    f1 = open(info_filepath, "a")
    f1.write(f'\n\nThe ID: {id}')
    f1.write(f'\nConcerning Final FSM')
    f1.write(f'\nThe similarity threshold: {args.sim_threshold}')
    f1.write(f'\nThe amount of all merging: {all_merge}')
    f1.write(f'\nThe amount of correct merging: {correct_merge}')


    day = date.today()
    os.makedirs(f"./Logs",exist_ok=True)
    checkout_path = f"./Logs/{day}.txt"
    """if not args.new_runtime :
        i=0
        while (os.path.isfile(checkout_path)):
            checkout_path = f"./Logs/{day}_({i}).txt"
            i += 1"""
                

    f1 = open(checkout_path, "a")
    f1.write(f'{id} | {len(dev_corpus)} words | {dev_accuracy}\n')
