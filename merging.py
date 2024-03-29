from copy import deepcopy
from scoring import *
from similarity import cosine
import numpy as np

def cosine_merging(fsm, states, threshold):
     
    all_merges, correct_merges = 0, 0
    fsm_ = deepcopy(fsm)
    
    sim1 = []
    for i in range(len(states)):
        sim1.append([])
        for j in range(len(states)):
            sim1[i].append(cosine(states[i], states[j]))

    #blockPrint()
    similarity_bool = []
    for i in range(len(states)):
        similarity_bool.append([])
        for j in range(len(states)):
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
                all_merges += x
                correct_merges += y
                
    
    fsm_.removeDuplicate()
    fsm_.id = str(fsm_.id) + 'min'
    return fsm_, all_merges, correct_merges

def cross_validate(left, right, fsm, states, states_mask, val_sents, val_gold):

    max_acc = -1

    for j in np.arange(left, right, .05):
        _fsm = deepcopy(fsm)
        merged_fsm, all_merges, correct_merges = cosine_merging(_fsm, states, j)
        cur_acc = score_all_prefixes(merged_fsm, val_sents, val_gold)
        # print(f'{j}\t{cur_acc}')
        if (cur_acc > max_acc):
            max_acc = cur_acc
            opt_threshold = j
            opt_fsm = deepcopy(merged_fsm)

    return opt_fsm, all_merges, correct_merges, opt_threshold, max_acc