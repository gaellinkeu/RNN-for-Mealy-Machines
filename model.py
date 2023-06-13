from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Embedding
import json
import numpy as np

def Tagger(n_tokens = 4, embedding_vector_length = 10, hidden_dim = 10, n_labels = 3, return_states = False):
    model = Sequential()
    model.add(Embedding(n_tokens, embedding_vector_length))
    model.add(SimpleRNN(hidden_dim, return_sequences=True))
    if not return_states:
        model.add(Dense(n_labels, activation='softmax'))
    return model

# These two functions work only for the specified configuration of the network
"""def save_weights(model, filename):
    weights = []
    for layer in model.layers:
        w = []
        [w.append(x.tolist()) for x in layer.get_weights()]
        weights.append(w)
    
    with open(filename, 'w') as f:
        json.dump(weights, f)"""

def load_weights(model, filename):
    with open(filename, 'r') as f:
        weights = json.load(f)
    
    """model.layers[0].set_weights([np.array(weights[0][0])])
    model.layers[1].set_weights([np.array(weights[1][0]), np.array(weights[1][1]), np.array(weights[1][2])])
    model.layers[2].set_weights([np.array(weights[2][0]), np.array(weights[2][1])])"""