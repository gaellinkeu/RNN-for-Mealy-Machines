import numpy as np
from copy import deepcopy

def preprocessing(inputs, outputs):

    max_length = len(max(inputs, key=len))
    x_train, y_train, pad_character = padding(inputs, outputs, max_length)

    input_alphabet = set([])
    output_alphabet = set([])
    for (x,y) in zip(x_train,y_train):
        input_alphabet = set(list(input_alphabet)+list(x))
        output_alphabet = set(list(output_alphabet)+list(y))

    X = np.array([one_hot_encoding(tokenization(x), len(input_alphabet)) for x in x_train])
    # X = np.array([(tokenization(x)) for x in x_train])

    Y = np.array([class_mapping(x, len(output_alphabet)) for x in y_train])
    mask = np.array([masking(x, pad_character) for x in x_train])

    return X, Y, len(input_alphabet), len(output_alphabet), mask


def padding(inputs, outputs, max_length):

    input_alphabet = set([])
    output_alphabet = set([])
    for (x,y) in zip(inputs,outputs):
        input_alphabet = set(list(input_alphabet)+list(x))
        output_alphabet = set(list(output_alphabet)+list(y))

    bos = ['e', 'f', 'g']  # Plausible beginning of sentence marker
    for b in bos:
        if b not in input_alphabet:
            break

    pad_characters = ['z', 'y', 'x'] # Plausible end of sentence marker
    for pad_character in pad_characters:
        if pad_character not in input_alphabet:
            break
     
    pad_inputs = [b+x+pad_character*(max_length-len(x)) for x in inputs]

    output_pads = ['0', '1', '2', '3', '4'] 
    for p in output_pads:
        if p not in output_alphabet:
            break
    p = list(output_alphabet)[0]
    pad_outputs = ['0'+x+p*(max_length-len(x)) for x in outputs]

    return pad_inputs, pad_outputs, pad_character


def class_mapping(label, numb_class = 2):
    y_train = []
    for x in label:
        assert int(x) < numb_class
        y_train.append([int(i==int(x)) for i in range(numb_class)])
        
    return y_train

# this function is specific to the input alphabet ['e', 'a', 'b', 'z', 'f', 'g', 'y', 'x']
def tokenization(word):
    x_train = []
    alphabet = ['e', 'a', 'b', 'z', 'f', 'g', 'y', 'x']
    for x in word:
        x_train.append(list(alphabet).index(x))
    
    return x_train

def one_hot_encoding(input, vocab_size=4):
    encoded_input = []
    for x in input:
        temp = [0.0]*vocab_size
        temp[x] = 1.0
        encoded_input.append(temp)
        
    return encoded_input

def one_hot_decoding(input):
    decoded_input = []
    
    for x in list(input):
        decoded_input.append(list(x).index(1))
    return decoded_input


def masking(word, pad_char = 'z'):
    return [x!=pad_char for x in word]