from keras import backend as K
# Accuracy ne prenant pas en compte les charactères complétés

# Remove this when the cosineSimilarity will be added


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


    max_length = 0

    for line in lines:
        res = ""
        isInput = True
        for symbol in line:
            if symbol in [',', '\n']:
                if isInput:
                    inputs.append(res)
                    max_length = len(res) if len(res) > max_length else max_length
                    res = ""
                    isInput = not isInput
                    continue
                else:
                    outputs.append(res)
            res += symbol
        #print(line)
    return inputs, outputs, max_length


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