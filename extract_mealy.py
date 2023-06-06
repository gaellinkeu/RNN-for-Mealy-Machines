import argparse
import time
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
from model2 import Tagger, load_weights
import pickle
#from IPython.display import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_length", type=int, default=10)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=300)
    parser.add_argument("--sim_threshold", type=float, default=.75)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--hidden_size", type=float, default=10)
    parser.add_argument("--fst", dest='fst', action='store_true')
    parser.set_defaults(fst=False)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.set_defaults(min=False)
    parser.add_argument('--eval', type=str, default="preds")
    parser.add_argument("--no_state_count", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--find_threshold", action="store_false")
    parser.add_argument("--symmetric_merging", action="store_true")
    parser.add_argument("--nondeterminism", action="store_true")
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

def cosine_merging(fsm, states, states_mask, threshold, symmetric=False):
    cos = tf.keras.losses.CosineSimilarity(axis=-1)
    #sim = -cos(states[None, :, :], states[:, None, :])
     
    total, pruned = 0, 0
    fsm_ = deepcopy(fsm)
    #print(tf.shape(sim.shape))
    #print(sim)
    fsm_.print(all = True)
    sim1 = []
    for i in range(len(states)):
        sim1.append([])
        for j in range(len(states)):
            sim1[i].append(cosine(states[i], states[j]))

    #blockPrint()
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
                total += 1
                
                res = fsm_.merge_states(j, i)
                #print(f'The number of states : {len(fsm_.nodes)}')
                
                #pruned += 1 - res
    #enablePrint()
    print(f'\nLe total des fusion est: {total}\n')
    fsm_.print()
    fsm_.removeDuplicate()
    fsm_.id = str(fsm_.id) + 'min'
    return fsm_

def cross_validate(left, right, fsm, states, states_mask, val_sents, val_gold, symmetric=False):

    max_acc = -1

    for j in np.arange(left, right, .05):
        _fsm = deepcopy(fsm)
        merged_fsm = cosine_merging(_fsm, states, states_mask, j, symmetric=False)
        cur_acc = score_all_prefixes(merged_fsm, val_sents, val_gold)
        if (cur_acc > max_acc):
            max_acc = cur_acc
            opt_threshold = j
            opt_fsm = deepcopy(merged_fsm)

    return opt_fsm, opt_threshold, max_acc


if __name__ == "__main__" :
    args = parse_args()

    print('\n\n\n'+'*'*20+' EXTRACTION OF MEALY MACHINE FROM RNN '+'*'*20+'\n\n\n')

    init_train_acc, init_dev_acc, train_acc, dev_acc = {}, {}, {}, {}
    train_acc["0"] = []
    n_train = range(1)
    id = args.id
    fsm_filepath = f'./FSMs/fsm{id}.txt'
    expected_fsm = getFsm(fsm_filepath)

    data_filepath = f'./datasets/dataset{id}.txt'

    for n in n_train:
        
        corpus, labels = get_data(data_filepath)
        corpus = corpus[:20]
        labels = labels[:20]
    
        """corpus = [x[:4] for x in corpus]
        labels = [x[:4] for x in labels]
        corpus, labels = corpus[:20], labels[:20]"""

        #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb', 'abb', 'aaaa', 'baaaaab', 'abababa']
        #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11', '101', '1000', '1100000', '1010101']
        assert(len(corpus) == len(labels))
        max_length = len(max(corpus, key=len))

        print(f'The corpus is {corpus[:5]}')
        print(f'The labels are {labels[:5]}')

        val_size = int(0.3 * len(corpus))
        val_corpus = labels[len(corpus) - val_size:]
        val_labels = labels[len(labels) - val_size:]

        corpus = corpus[:len(corpus) - val_size]
        labels = labels[:len(labels) - val_size]
        


        dev_size = int(0.2 * len(corpus))
        dev_corpus = corpus[len(corpus) - dev_size:]
        dev_labels = labels[len(labels) - dev_size:]

        corpus = corpus[:len(corpus) - dev_size]
        labels = labels[:len(labels) - dev_size]
        corpus_, labels_ = preprocessing(corpus, labels, max_length)
        #corpus_ = ["e"+x+"z"*(max_length-len(x)) for x in corpus]
        #labels_ = ["0"+x+"2"*(max_length-len(x)) for x in labels]
        labels__ = np.array([np.array(list(x)) for x in labels_])
        mask = [x!="z" for x in corpus_]

        
        #print(corpus_)
        #print(states)

        x_train = np.array([tokenization(x) for x in corpus_])
        train_sents = [tokenization(x) for x in corpus]
        y_train = np.array([class_mapping(x) for x in labels_])
        mask = np.array([masking(x) for x in corpus_])

        print('\n\033[FData Preprocessing... Done\n')
        trained_model = Tagger(4, 10, args.hidden_size, 3)
        trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        filename = f'./weights/weights{id}.txt'
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        print('\033[FModel definition... Done\n')
        trained_model.set_weights(weights)
        print('\033[FModel Update... Done\n')
        #load_weights(trained_model, 'weights.txt')
    
        redundant_fsm = build_fsm_from_dict(n, corpus, labels)
        #redundant_fsm.print()
        print('\033[FTrie Building... Done\n')
        assert(score_all_prefixes(redundant_fsm, corpus, labels) == 100.0)
        print('\033[FChecking if the trie get the right ouput for each input... Done\n')
        trained_model.pop()
        
        representations = trained_model.predict(x_train)
        print('\033[FGetting states... Done\n')
        idx = [redundant_fsm.return_states(sent) for sent in corpus] # maps strings to states
        n_states = len(redundant_fsm.nodes)
        states = np.zeros((n_states, args.hidden_size))
        states_mask = np.zeros(n_states)
        print('\033[FStates Mapping preparation... Done\n')
        
        #print(idx)

        for i, _r in enumerate(representations):
            states[idx[i]] = _r[mask[i]]
            states_mask[idx[i]] = labels__[i][mask[i]]
        #print(states)
        #print(states[0])
        #print(states[1])
        print('\033[FStates Mapping... Done\n')
        init_fsm = deepcopy(redundant_fsm)
        find_threshold = False
        print('\033[FMerging Preparation... Done\n')

        if(find_threshold):
            # code for finding the good similarity threshold
            print("We are findinf the optimal threshold")
            merged_fsm, _, _ = cross_validate(.5, 1., redundant_fsm, states, states_mask, val_corpus, val_labels)
            args.find_threshold = True
        else:
            merged_fsm = cosine_merging(redundant_fsm, states, states_mask, threshold=args.sim_threshold, symmetric=args.symmetric_merging)
        print('\033[FMerging stage... Done\n')
        merged_fsm.print(all=True)
    
    # Checking the equivalence between the get
    equivalence = fsm_equivalence(expected_fsm, merged_fsm)

    if equivalence:
        print('\n\nThe obtained FSM is EQUIVALENT to the one we expected\n')
    else:
        print('\n\nThe obtained FSM is NOT EQUIVALENT to the expected FSM\n')
    
    _acc = score_whole_words(merged_fsm, dev_corpus, dev_labels)
    #_acc = 0
    print('\033[FGetting the accuracy... Done\n')
    train_acc["0"].append(_acc)
    
    print('\n****************  WE HAVE FINISHED   ***************')
    print(f'\n*************   THE ACCURACY IS :   {_acc}   *****************')