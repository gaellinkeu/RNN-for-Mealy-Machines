import argparse
from model import Tagger
from utils import *
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_length", type=int, default=10)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=300)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
  
    args = parse_args()

    id = args.id

    print('\n\n\n'+'*'*20+f' ID {id}: '+' TRAINING THE RECURRENT NEURAL NETWORK '+'*'*20+'\n\n\n')
    #max_length = 4
    #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus, labels = get_data(f'./datasets/dataset{id}.txt')


    max_length = len(max(corpus, key=len))

    print('The five first elements are')
    print(f'Corpus : {corpus[:5]}')
    print(f'Labels : {labels[:5]}')

    """dev_size = int(args.dev_percentage * len(corpus))
    dev_corpus = corpus[len(corpus) - dev_size:]
    dev_labels = labels[len(corpus) - dev_size:]

    corpus = corpus[:len(corpus) - dev_size]
    labels = labels[:len(labels) - dev_size]"""
    corpus_, labels_ = preprocessing(corpus, labels, max_length)

    
    """corpus_ = ["e"+x+"z"*(max_length - len(x)) for x in corpus]
    labels_ = ["0"+x+"2"*(max_length - len(x)) for x in labels]"""
    print(f'\nThe length of corpus is: {len(corpus)}\n')

    x_train = np.array([tokenization(x) for x in corpus_])
    y_train = np.array([class_mapping(x) for x in labels_])
    mask = np.array([masking(x) for x in corpus_])

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=args.test_size, random_state=42)
    """version_name = '01'
    model_dir = os.path.join("weigths", version_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = "weigths/model_weights.h5\""""

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model = Tagger(4, 10, 10, 3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, args.batch_size, args.n_epochs)
    # bacth de taille 2
    
    loss = history.history['loss'][-1]
    accuracy = history.history['accuracy'][-1]
    #print('\n\n\n Les prédictions sont: \n\n')
    #print(train_preds)
    
    print(f'\n The training categorical crossentropy loss: {loss}')
    print(f'\n The training accuracy: {accuracy*100} %')

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("\n The testing accuracy: %.2f%%" % (scores[1]*100))

    os.makedirs(f"./Infos",exist_ok=True)
    info_filepath = f'./Infos/Execution {id}.txt'
    f1 = open(info_filepath, "a")
    f1.write(f'The ID: {id}')
    f1.write(f'\nConcerning RNN: {id}')
    f1.write(f'\nThe batch size: {args.batch_size}')
    f1.write(f'\nThe amount of epoch: {args.n_epochs}')
    f1.write(f'\nThe training dataset size: {x_train.shape[0]}')
    f1.write(f'\nThe testing dataset size: {x_test.shape[0]}')
    f1.write(f'\nThe training accuracy: {accuracy*100} %\n')
    f1.write("The testing accuracy: %.2f%%" % (scores[1]*100))

    os.makedirs(f"./weights",exist_ok=True)
    filename = f'./weights/weights{id}.txt'
    with open(filename, 'wb') as f:
        pickle.dump(model.get_weights(), f)
