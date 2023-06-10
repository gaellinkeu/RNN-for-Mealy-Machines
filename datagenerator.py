# author : Omer Nguena Timo
# version : v0
# use : research only

import os
import random
from fsm import FSM
from state import State
import argparse

def randomWord(min_length, max_length, vocab) -> str:
    word = ""
    length = random.randrange(min_length, max_length)
    for i in range(length):
        word += random.choice(list(vocab))
    return word

def datasetSaving(id, inputs, outputs):
    assert len(inputs) == len(outputs), 'The inputs set and output sets don\'t have the same length'
    os.makedirs(f"./datasets", exist_ok=True)
    f = open(f"./datasets/dataset{id}.txt", "w")
    for i in range(len(inputs)):
        f.write(f'{inputs[i]},{outputs[i]}\n')
    f.close()

        
def buildExampleFsm0() -> FSM :
   fsm = FSM()
   states=[State(0), State(1)]
   transitions = [(0, 0, "b", "1"), (0, 1, "a", "1"), (1, 1, "a", "0"), (1, 0, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
      fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   #for i in fsm._statesById.keys():
        #print(f's: {fsm.getState(state)}')
        #fsm._statesById[i].print()
        #pass

   return fsm

def buildExampleFsm1() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2)]
   transitions = [(0, 1, "b", "1"), (0, 0, "a", "0"), (1, 1, "a", "1"), (1, 2, "b", "0"), (2, 0, "b", "0"), (2, 1, "a", "1")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
      fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   """for i in fsm._statesById.keys():
        #print(f's: {fsm.getState(state)}')
        fsm._statesById[i].print()"""

   return fsm

def buildExampleFsm2() -> FSM :
   fsm = FSM()
   states=[State(0), State(1), State(2), State(3)]
   transitions = [(0, 1, "b", "1"),
                (0, 3, "a", "1"),
                (1, 1, "a", "1"),
                (1, 2, "b", "1"),
                (2, 0, "a", "0"),
                (2, 3, "b", "0"),
                (3, 1, "a", "0"),
                (3, 3, "b", "0")]
   for state in states :
      fsm.addState(state)
   for transition in transitions:
       fsm.addTransition(transition[0], transition[1], transition[2], transition[3])

   """for i in fsm._statesById.keys():
        #print(f's: {fsm.getState(state)}')
        fsm._statesById[i].print()"""

   return fsm



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
    parser.add_argument("--train_length", type=int, default=1000)
    parser.add_argument("--n_train_low", type=int, default=1)
    parser.add_argument("--n_train_high", type=int, default=12)
    parser.add_argument("--n_states", type=int, default=5)
    return parser.parse_args()



if __name__ == '__main__' :
   #cwd = os.getcwd()  # Get the current working directory (cwd)
   #files = os.listdir(cwd)  # Get all the files in that directory
   #print("Files in %r: %s" % (cwd, files))
    args = parse_args()

    id = args.id
    print('\n\n\n'+'*'*20+f' ID {id}: '+' RANDOM FSM INITIALIZING AND DATA GENERATION '+'*'*20+'\n\n\n')

    N = args.train_length
    max_length = args.n_train_high
    min_length = args.n_train_low
    n_states = args.n_states
    
    if id == 0:
        fsm = buildExampleFsm0()
    elif id == 1:
        fsm = buildExampleFsm1()
    elif id == 2:
        fsm = buildExampleFsm2()
    else:
        fsm = fsmRandomGenInputComplete(n_states)

    
    #print(F.produceOutput('abbaba'))

    # The first FSM
    inputs_set = []
    outputs_set = []

    for _ in range(N):
        word = randomWord(min_length, max_length, fsm._inputSet)
        inputs_set.append(word)
        outputs_set.append(fsm.produceOutput(word))
        #print(f"{word} => {fsm.produceOutput(word)}\n")
    datasetSaving(id, inputs_set, outputs_set)
    fsm.print()
    fsm.save(id)

    """
    for k in range(1, 2):
        nbState = random.randrange(6,15)
        fsm = fsmRandomGenInputComplete(nbState)
        os.makedirs(f"./data/exemple{k}",exist_ok=True)
        f = open(f"./data/exemple{k}/fsm.dot", "w")
        f.write(fsm.toDot())
        f.close()"""