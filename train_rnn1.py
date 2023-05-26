from model2 import Tagger
from utils import get_data, class_mapping, tokenization, masking, ignore_class_accuracy
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
import pickle

if __name__ == "__main__":
  
    #max_length = 4
    #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus, labels, max_length = get_data('dataset2.txt')
    corpus_ = ["e"+x+"z"*(max_length-len(x)) for x in corpus]
    labels_ = ["0"+x+"2"*(max_length - len(x)) for x in labels]
    states = []

    n_epochs = 5
    
    batch_size = 100

    x_train = np.array([tokenization(x) for x in corpus_[:100]])
    y_train = np.array([class_mapping(x) for x in labels_[:100]])
    mask = np.array([masking(x) for x in corpus_[:101]])

    """version_name = '01'
    model_dir = os.path.join("weigths", version_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = "weigths/model_weights.h5\""""

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model = Tagger(4, 10, 10, 3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', ignore_class_accuracy(2)])

    history = model.fit(x_train, y_train, batch_size, n_epochs)
    # bacth de taille 2
    
    loss = history.history['loss'][-1]
    accuracy = history.history['accuracy'][-1]
    #print('\n\n\n Les prédictions sont: \n\n')
    #print(train_preds)
    
    print(f'\n\n The loss: {loss}')
    print(f'\n\n The accuracy: {accuracy}')

    filename = 'weights.txt'
    with open(filename, 'wb') as f:
        pickle.dump(model.get_weights(), f)
