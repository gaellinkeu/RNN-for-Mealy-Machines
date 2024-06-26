# author : Omer Nguena Timo
# Modified: Gaël LINKEU
# version : v0
# use : research only

import os
import random
from fsm import FSM
from state import State
import argparse
from utils import randomWord, n_uplets
from defined_machines import *

# Dataset saving
def datasetSaving(id, inputs, outputs):
    assert len(inputs) == len(outputs), 'The inputs set and output sets don\'t have the same length'
    os.makedirs(f"./datasets", exist_ok=True)
    f = open(f"./datasets/dataset{id}.txt", "w")
    for i in range(len(inputs)):
        f.write(f'{inputs[i]},{outputs[i]}\n')
    f.close()

# Random generation of a new Mealy Machine
def fsmRandomGenInputComplete(nbStates=2,inputAlphabet =['a','b'], outputAlphabet =['0','1']) -> FSM :
    fsm = FSM()
    maxNbTransition = nbStates *  len(inputAlphabet)
    stateIds = [i for i in range(0,nbStates)]
    for i in stateIds :
      fsm.addState(State(i))
    fin = (fsm.nbTransitions()>=maxNbTransition) 
    while not(fin) :
        idSrcState = random.choice(stateIds)
        idTgtState = random.choice(stateIds)
        input = random.choice(inputAlphabet)
        if not (fsm.getState(idSrcState).defineTransitionOn(input)):
            output = random.choice(outputAlphabet)
            tr = fsm.addTransition(idSrcState,idTgtState,input,output)
        fin = (fsm.nbTransitions()>=maxNbTransition)
    return fsm  
    

# Parameters initialization
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=2)
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--min_word_length", type=int, default=1)
    parser.add_argument("--max_word_length", type=int, default=8)
    parser.add_argument("--n_states", type=int, default=None)
    parser.add_argument("--static", type=int, default=1)
    parser.add_argument("--complete_dataset", type=int, default=0)
    return parser.parse_args()



if __name__ == '__main__' :
    args = parse_args()

    id = args.id
    print('\n\n\n'+'*'*20+f' ID {id}: '+' RANDOM FSM INITIALIZATION AND DATA GENERATION '+'*'*20+'\n\n\n')

    N = args.dataset_size
    max_length = args.max_word_length
    min_length = args.min_word_length
    if args.n_states == None:
        # The number of state should be twice(the length of our alphabet)
        # the id of the machine
        n_states = id+2
    else:
        n_states = args.n_states
    
    if not args.static:
        # Random generation
        fsm = fsmRandomGenInputComplete(n_states)
    else:
        # We get predefined Mealy Machine
        if id == 0:
            fsm = buildExampleFsm0()
        if id == 1:
            fsm = buildExampleFsm1()
        if id == 2:
            fsm = buildExampleFsm2()
        if id == 3:
            fsm = buildExampleFsm3()
        if id == 4:
            fsm = buildExampleFsm4()
        if id == 5:
            fsm = buildExampleFsm5()
        if id == 6:
            fsm = buildExampleFsm6()
        if id == 7:
            fsm = buildExampleFsm7()
        if id == 8:
            fsm = buildExampleFsm8()
        if id == 9:
            fsm = buildExampleFsm9()
    
    #print(F.produceOutput('abbaba'))

    # The first FSM
    inputs_set = []
    outputs_set = []

    # Dataset Generation

    if args.complete_dataset :
        inputs_set = n_uplets(fsm._inputSet, max_length)
        for word in inputs_set:
            outputs_set.append(fsm.produceOutput(word))
    else:
        for _ in range(N):
            word = randomWord(min_length, max_length, fsm._inputSet)
            inputs_set.append(word)
            outputs_set.append(fsm.produceOutput(word))

    datasetSaving(id, inputs_set, outputs_set)
    fsm.print()
    fsm.save(id)