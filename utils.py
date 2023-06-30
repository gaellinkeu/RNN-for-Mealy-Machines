from keras import backend as K
import sys, os
from mealy_machine import *
import random
# Accuracy ne prenant pas en compte les charactères complétés

# Remove this when the cosineSimilarity will be added

def randomWord(min_length, max_length, vocab) -> str:
    word = ""
    length = random.randrange(min_length, max_length)
    for i in range(length):
        word += random.choice(list(vocab))
    return word


def ignore_class_accuracy(to_ignore=2):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

# Fonction qui imitte le comportement du réseau de neurone
def get_hidden_state (word):
    prec = 0.005
    output = []
    
    for i, char in enumerate(word):
        if char == "a":
            prec = 0.02*(i+1) + prec
            output.append(prec)
        elif char == "b":
            prec = 0.03*(i+1) + prec
            output.append(prec)
        elif char == "e":
            prec = 0.005*(i+1) + prec
            output.append(prec)
        else:
            prec = 0.05*(i+1) + prec
            output.append(prec)
    return output



def get_data(filepath):

    inputs = []
    outputs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    #print(lines[:3])


    #max_length = 0

    for line in lines:
        res = ""
        isInput = True
        for symbol in line:
            if symbol in [',', '\n']:
                if isInput:
                    inputs.append(res)
                    #max_length = len(res) if len(res) > max_length else max_length
                    res = ""
                    isInput = not isInput
                    continue
                else:
                    outputs.append(res)
            res += symbol
        #print(line)
    return inputs, outputs

def preprocessing(corpus, labels, max_length):
    
    bos = ['e', 'f', 'g']  # Plausible beginning of sentence marker
    for b in bos:
        if b not in set(corpus):
            break

    eos = ['z', 'y', 'x'] # Plausible end of sentence marker
    
    for e in eos:
        if e not in set(corpus):
            break
    pad_label = ['0', '1', '2', '3', '4']  

    corpus_ = [b+x+e*(max_length-len(x)) for x in corpus]

    for p in pad_label:
        if p not in set(corpus):
            break
    labels_ = ['0'+x+p*(max_length-len(x)) for x in labels]
    return corpus_, labels_


def class_mapping(label, numb_class = 3):
    y_train = []
    for x in label:
        assert int(x) < numb_class
        y_train.append([int(i==int(x)) for i in range(numb_class)])
        
    return y_train

def tokenization(word, num_token = 3):
    x_train = []
    for x in word:
        if x == 'a':
            x_train.append(0)
        elif x == 'b':
            x_train.append(1)
        elif x == 'e':
            x_train.append(2)
        else:
            x_train.append(3)
    
    return x_train

def masking(word, pad_char = 'z'):
    return [x!=pad_char for x in word]


def ignore_class_accuracy(to_ignore=2):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

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



# Compute the cosine similarity between two vectors
def cosine(h1, h2):
    cos = 0
    s1 = 0
    s2 = 0
    assert len(h1) == len(h2)
    for i in range(len(h1)):
        cos += h1[i]*h2[i]
        s1 += h1[i]**2
        s2 += h2[i]**2
    s1 = s1**(1/2)
    s2 = s2**(1/2)
    return cos/(s1*s2)

# Compute the eucledian distance between 
def euclidian(h1, h2):
    assert len(h1) == len(h2)
    distance = 0
    for i in range(len(h1)):
        distance += (h1[i] - h2[i])**2
    distance = distance**(1/2)
    return distance

# Check the equivalence between two fsm
def fsm_equivalence(fsm1 : Mealy, fsm2 : Mealy):
    if set(fsm1.inputAlphabet) != set(fsm2.inputAlphabet):
        print('The two FSM don\'t have the same input set' )
        return 0
    if set(fsm1.outputAlphabet) != set(fsm2.outputAlphabet):
        print('The two FSM don\'t have the same output set' )
        return 0
    start = (fsm1.root, fsm2.root)
    inputAlphabet = set(fsm1.inputAlphabet + fsm2.inputAlphabet)
    outputAlphabet = set(fsm1.outputAlphabet + fsm2.outputAlphabet)
    nodes = [start]
    nodes_ = deepcopy(nodes)
    #print('Starting')
    #print(nodes[0])
    while len(nodes) != 0:
        node = nodes[0]
        for x in inputAlphabet:
            output1 = fsm1.output(node[0], x)
            output2 = fsm2.output(node[1], x)
            #print(f'The len is {len(nodes)}')
            #print(f'{node}   {x}   {(output1[1],output2[1])}')
            if (output1 == None and output2!=None) or (output1 != None and output2==None):
                return 0
            if output1[0] != output2[0]:
                return 0
            
            #print((output1[1], output2[1]))
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
    #print(arcs)
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


    """while i < len(arcs_):
        
        arc = (int(arcs_[i]), arcs_[i+1], arcs_[i+2], int(arcs_[i+3]))
        print(arc)
        arcs.append(arc)
        i = i+4"""
    
    fsm = Mealy(id, states[0], states, arcs)
    return fsm

# Block all the print calls
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore all the print calls
def enablePrint():
    sys.stdout = sys.__stdout__