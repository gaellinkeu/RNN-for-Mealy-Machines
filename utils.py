from keras import backend as K
import sys, os
from mealy_machine import *
import random

def random_selection(data, size):
    index = []
    output = []
    while len(index) < size:
        n = random.randint(0, len(data)-1)
        if n not in index:
            index.append(n)
    for i in index:
        output.append(data[i])
    return output

def train_test_splitting(data, labels, percent = False, train_size = 100, val_size = 0):
    #val_size = 2 * train_size
    train_corpus, train_labels, val_corpus, val_labels = [], [], [], []
    
    if percent:
        train_size = int(train_size * len(data))
        val_size = int(val_size * len(data))
    #exit(0)
    index = []
    while len(index) < train_size:
        n = random.randint(0, len(data)-1)
        if n not in index:
            index.append(n)
    for i in index:
        train_corpus.append(data[i])
        train_labels.append(labels[i])

    index2 = []
    while len(index2) < val_size:
        n = random.randint(0, len(data)-1)
        if n in index or n in index2: 
            continue
        index2.append(n)
    for i in index2:
        val_corpus.append(data[i])
        val_labels.append(labels[i])

    return train_corpus, train_labels, val_corpus, val_labels


def randomWord(min_length, max_length, vocab) -> str:
    word = ""
    length = random.randrange(min_length, max_length)
    for _ in range(length):
        word += random.choice(list(vocab))
    return word

def n_uplets(vocab, word_length):
    inputs = []
    while(len(inputs) < (len(vocab) ** word_length)):
        word = ""
        for _ in range(word_length):
            word += random.choice(list(vocab))
        # word = [random.choice(vocab) for _ in range(word_length)]
        if word not in inputs:
            inputs.append(word)
    return inputs


def list_to_string(list_, mask):
    string = ''
    for i, x in enumerate(list_):
        if i == 0:
            continue
        if mask[i]:
            string += f'{x}'
    return string

def nparray_to_string(predictions, mask):
    preds = predictions.tolist()
    labels = []
    for i, x in enumerate(preds):
        labels.append(list_to_string(x, mask[i]))

    return labels


# Check the equivalence between two fsm
def fsm_equivalence(fsm1 : Mealy, fsm2 : Mealy):
    """if set(fsm1.inputAlphabet) != set(fsm2.inputAlphabet):
        print('The two FSM don\'t have the same input set' )
        return 0
    if set(fsm1.outputAlphabet) != set(fsm2.outputAlphabet):
        print('The two FSM don\'t have the same output set' )
        return 0"""
    start = (fsm1.root, fsm2.root)
    inputAlphabet = set(fsm1.inputAlphabet + fsm2.inputAlphabet)
    outputAlphabet = set(fsm1.outputAlphabet + fsm2.outputAlphabet)
    nodes = [start]
    nodes_ = deepcopy(nodes)
    
    while len(nodes) != 0:
        node = nodes[0]
        for x in inputAlphabet:
            output1 = fsm1.output(node[0], x)
            output2 = fsm2.output(node[1], x)
            
            if (output1 == None and output2!=None) or (output1 != None and output2==None):
                return 0
            if output1[0] != output2[0]:
                return 0
            
            node_ = (output1[1], output2[1])
            if node_ not in nodes_:
                nodes_.append(node_)
                nodes.append(node_)
        
        nodes.pop(0)
    return 1
        
def getFsm(filepath):
    removed_chars = [' ', "'", '\n']
    removed_chars2 = [' ', '[', ']', "'", '\n']
    with open(filepath, 'r') as f:
        lines = f.readlines()
    id = int(lines[0][0])
    states_ = list(lines[1])
    states_ = list(filter(lambda x: x not in removed_chars, states_))
    
    states = []
    word = ''
    start = True
    for x in states_:
        if x == '[':
            start = True
            continue
        if x == ']':
            states.append(int(word))
            start=False
        if start:
            if x==',':
                states.append(int(word))
                word=''
            else:
                word += x

    states = [int(x) for x in states]

    arcs = list(lines[2])
    arcs_ = list(filter(lambda x: x not in removed_chars2, arcs))
    arcs = []
    arc = []
    i = 0
    start=False
    word = ''
    for x in arcs_:
        if x=='(':
            start=True
            continue
        if x==')':
            arc.append(word)
            word=''
            arcs.append((int(arc[0]),arc[1],arc[2],int(arc[3])))
            arc= []
            start=False
            continue
        
        if start:
            if x==',':
                arc.append(word)
                word=''
            else:
                word += x

    fsm = Mealy(id, states[0], states, arcs)
    return fsm

# Block all the print calls
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore all the print calls
def enablePrint():
    sys.stdout = sys.__stdout__