import argparse
from model import Tagger
from utils import *
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow import keras
#from keras import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=300)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=30)
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
    #max_length = 4
    #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus, labels = get_data(f'./datasets/dataset{id}.txt')

    max_length = len(max(corpus, key=len))

    print('The five first elements are')
    print(f'Corpus : {corpus[:5]}')
    print(f'Labels : {labels[:5]}')

    # Data Preprocessing
    corpus_, labels_ = preprocessing(corpus, labels, max_length)

    
    print(f'\nThe length of corpus is: {len(corpus)}\n')
    # we continue the preprocessing
    x_train = np.array([tokenization(x) for x in corpus_])
    y_train = np.array([class_mapping(x) for x in labels_])
    mask = np.array([masking(x) for x in corpus_])

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=args.test_size, random_state=42)
    

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Initialization of the model
    model = Tagger(4, 10, 10, 3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, args.batch_size, n_epochs, callbacks=early_stopping)


    # Saving weights
    os.makedirs(f"./weights",exist_ok=True)
    filename = f'./weights/weights{id}.txt'
    with open(filename, 'wb') as f:
        pickle.dump(model.get_weights(), f)
    
    loss = history.history['loss'][-1]
    accuracy = history.history['accuracy'][-1]

    # Getting the epoch where we achieved 100 accuracy for the first time
    epoch_convergence = 0
    for i in range(len(history.history['accuracy'])):
        if int(history.history['accuracy'][i]*100) == 100:
            epoch_convergence = i+1
            break
        
    #print('\n\n\n Les prédictions sont: \n\n')
    #print(train_preds)
    
    print(f'\n The training categorical crossentropy loss: {loss}')
    print(f'\n The training accuracy: {accuracy*100} %')

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("\n The testing accuracy: %.2f%%" % (scores[1]*100))

    os.makedirs(f"./Results/{id}",exist_ok=True)
    results_filepath = f'./Results/{id}/rnn_training.txt'
    
    # Results Saving
    f1 = open(results_filepath, 'a+')
    lines = f1.readlines()
    if os.path.getsize(results_filepath) == 0:
        f1.write('ID,Time,Train_set_size,Test_set_size,convergence_epoch,training_acc,testing_acc')
        f1.write(f'\n{id},{args.times},{x_train.shape[0]},{x_test.shape[0]},{epoch_convergence},{accuracy*100},{scores[1]*100}')
    else:
        f1.write(f'\n{id},{args.times},{x_train.shape[0]},{x_test.shape[0]},{epoch_convergence},{accuracy*100},{scores[1]*100}')
    f1.close()
