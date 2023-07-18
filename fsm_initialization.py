# author : Omer Nguena Timo
# version : v0
# use : research only

import os
import random
from fsm import FSM
from state import State
import argparse
from utils import randomWord
from defined_machines import *


def datasetSaving(id, inputs, outputs, times=0):
    assert len(inputs) == len(outputs), 'The inputs set and output sets don\'t have the same length'
    os.makedirs(f"./datasets", exist_ok=True)
    f = open(f"./datasets/dataset{id}_{times}.txt", "w")
    for i in range(len(inputs)):
        f.write(f'{inputs[i]},{outputs[i]}\n')
    f.close()


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
            #print(tr.toDot()) 
        fin = (fsm.nbTransitions()>=maxNbTransition)  
    #print(f'The length: {len(fsm._statesById)}')
    #print(fsm.toDot())
    return fsm  
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=-1)
    parser.add_argument("--train_length", type=int, default=10000)
    parser.add_argument("--n_train_low", type=int, default=1)
    parser.add_argument("--n_train_high", type=int, default=15)
    parser.add_argument("--n_states", type=int, default=2)
    parser.add_argument("--static", type=int, default=1)
    parser.add_argument("--times", type=int, default=0)
    return parser.parse_args()



if __name__ == '__main__' :
   #cwd = os.getcwd()  # Get the current working directory (cwd)
   #files = os.listdir(cwd)  # Get all the files in that directory
   #print("Files in %r: %s" % (cwd, files))
    args = parse_args()

    id = args.id
    print('\n\n\n'+'*'*20+f' ID {id} TIMES {args.times}: '+' RANDOM FSM INITIALIZING AND DATA GENERATION '+'*'*20+'\n\n\n')

    N = args.train_length
    max_length = args.n_train_high
    min_length = args.n_train_low
    n_states = args.n_states
    
    if not args.static:
        fsm = fsmRandomGenInputComplete(n_states)
    else:
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

    for _ in range(N):
        word = randomWord(min_length, max_length, fsm._inputSet)
        inputs_set.append(word)
        outputs_set.append(fsm.produceOutput(word))
        #print(f"{word} => {fsm.produceOutput(word)}\n")
    datasetSaving(id, inputs_set, outputs_set, args.times)
    fsm.print()
    fsm.save(id, args.times)

    """
    for k in range(1, 2):
        nbState = random.randrange(6,15)
        fsm = fsmRandomGenInputComplete(nbState)
        os.makedirs(f"./data/exemple{k}",exist_ok=True)
        f = open(f"./data/exemple{k}/fsm.dot", "w")
        f.write(fsm.toDot())
        f.close()"""