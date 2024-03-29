import argparse
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
#from keras import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle

from model import Tagger, save_weights
from utils import *
from training_data_preprocessing import preprocessing
from save_results import save_training_results
from loading_data import load_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=2)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=300)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--times", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
  
    args = parse_args()

    id = args.id
    """n_epochs = args.n_epochs

    if id > 2:
        n_epochs *= 2
    if id > 5:
        n_epochs *= 2 """
    n_epochs = args.n_epochs
    # The Training after 5 consecutive epochs and no evolution of the acc
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5, restore_best_weights=True)

    print('\n\n\n'+'*'*20+f' ID {id} TIMES {args.times}: '+' TRAINING THE RECURRENT NEURAL NETWORK '+'*'*20+'\n\n\n')
  
    # Data Loading
    inputs, outputs, max_length = load_data(f'./datasets/dataset{id}.txt')


    # Data Preprocessing
    X, Y, n_tokens, n_labels, _ = preprocessing(inputs, outputs)
  
    
    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.test_size, random_state=42)

    print('Parameters')
    print(f' * id : {args.id}\n * n_tokens: {n_tokens}\n * hidden_size: {args.hidden_size}\n * n_labels: {n_labels}\n')
    
    # Initialization of the model
    # model = Tagger(n_tokens, 10, args.hidden_size, n_labels)
    model = Tagger(n_tokens,max_length+1,args.hidden_size, n_labels=n_labels)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, args.batch_size, n_epochs, callbacks=early_stopping)

    
    # Analyze the model loss and accuracy
    loss = history.history['loss'][-1]
    acc = history.history['accuracy'][-1]
    
    print(f'\n The training categorical crossentropy loss: {loss}')
    print(f'\n The training accuracy: {acc*100} %')

    test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\n The testing accuracy: %.2f%%" % (test_acc[1]*100))
    dev_acc = model.evaluate(X, Y, verbose=0)
    print("\n The dev accuracy: %.2f%%" % (dev_acc[1]*100))

    # Save the model (saving weights)
    os.makedirs(f"./weights",exist_ok=True)
    filepath = f'./weights/weights{id}.txt'
    save_weights(model, filepath)

    os.makedirs(f"./Results/{id}",exist_ok=True)
    results_filepath = f'./Results/{id}/rnn_training.txt'
    
    # Results Saving
    save_training_results(results_filepath, args.id, args.times, x_train, x_test, acc, test_acc, dev_acc)
