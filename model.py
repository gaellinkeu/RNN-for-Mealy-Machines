from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Embedding, Input

import json
import numpy as np
import pickle
from utils import nparray_to_string
from copy import deepcopy

def Tagger(n_tokens = 3, max_length=3, embedding_vector_length = 10, hidden_dim = 10, n_labels = 2, return_states = False):
    model = Sequential()
    # model.add(Embedding(n_tokens, embedding_vector_length))
    model.add(Input(shape=(max_length,n_tokens)))
    model.add(SimpleRNN(hidden_dim, return_sequences=True))
    if not return_states:
        model.add(Dense(n_labels, activation='softmax'))
    return model

# These two functions work only for the specified configuration of the network
def save_weights(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model.get_weights(), f)

# def load_weights(model, filename):
#     with open(filename, 'r') as f:
#         weights = json.load(f)
    
    """model.layers[0].set_weights([np.array(weights[0][0])])
    model.layers[1].set_weights([np.array(weights[1][0]), np.array(weights[1][1]), np.array(weights[1][2])])
    model.layers[2].set_weights([np.array(weights[2][0]), np.array(weights[2][1])])"""



def load_model(id, n_tokens, max_length, n_embedding, hidden_size, n_labels):

    trained_model = Tagger(n_tokens, max_length+1, hidden_size, n_labels=n_labels)
    trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filename = f'./weights/weights{id}.txt'
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    
    # Retrieve model weights
    trained_model.set_weights(weights)

    return trained_model


def get_rnn_states_and_outputs(trained_model, x_train, mask):

    model = deepcopy(trained_model)
    predictions = model.predict(x_train)
    predictions = predictions.argmax(axis=-1)
    pred_outputs = nparray_to_string(predictions, mask)

    model.pop()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    states = model.predict(x_train)

    return states, pred_outputs

def ignore_class_accuracy(to_ignore=2):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy