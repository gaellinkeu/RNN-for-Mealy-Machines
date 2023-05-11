import argparse
from copy import deepcopy
from mealy_trie import *
from utils import *
import os
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from utils import *
import matplotlib.pyplot as plt
from model import Tagger
#from IPython.display import Image




if __name__ == '__main__' :
    corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus_ = [ "e"+x+"z"*(4-len(x)) for x in corpus]
    labels_ = ["0"+x+"2"*(4-len(x)) for x in labels]
    states = [get_hidden_state(x) for x in corpus_]
    #print(corpus_)
    #print(states)


    
    #Image("Mealy.png")
    arbre_prefix = Trie(corpus_, labels_)
    arbre_prefix.assign_state(corpus_)
    #arbre_prefix.print()

    """
    filepath = "weigths/model_weights.h5"
    model_filepath = "models/model1.h5"
    embedding_vector_length = 10

    model1 = Tagger(3, 10, 10)

    """
    """
    model1 = Sequential()
    model1.add(Embedding(3, embedding_vector_length, input_length=max))
    model1.add(LSTM(100, return_sequences=True))
    model1.add(Dense(3, activation='softmax'))
    # Compile model
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ignore_class_accuracy(2)])
    model1.load_weights(filepath)"""



    

    
