from model import Tagger 
from utils import get_data, class_mapping, tokenization, masking
import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
  
    #max_length = 4
    #corpus = ['ba', 'b', 'a', 'baa', 'a', 'baaa', 'aa', 'b', 'abaa', 'abb', 'bb']
    #labels = ['11', '1', '1', '110', '1', '1100', '10', '1', '1010', '101', '11']
    corpus, labels, max_length = get_data('dataset2.txt')
    corpus_ = ["e"+x+"z"*(max_length-len(x)) for x in corpus]
    labels_ = ["0"+x+"2"*(max_length - len(x)) for x in labels]
    states = []

    n_epochs = 10
    batch_size = 10

    x_train = np.array([tokenization(x) for x in corpus_])
    y_train = np.array([class_mapping(x) for x in labels_])
    mask = np.array([masking(x) for x in corpus_])

    version_name = '01'
    model_dir = os.path.join("weigths", version_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = "weigths/model_weights.h5"

    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    trained_model = Tagger(4, 10, max_length+1, 10)
    """trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #trained_model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    train_results = trained_model(x_train, y_train, mask)"""

    for epoch in range(5):
        for batch_idx in trange(0, len(x_train) - 2, 2):
            with tf.GradientTape() as tape:
                batch_tokens = x_train[batch_idx:batch_idx + 2]
                batch_labels = y_train[batch_idx:batch_idx + 2]
                batch_mask = mask[batch_idx:batch_idx + 2]
                train_results = trained_model(batch_tokens, batch_labels, batch_mask)
                
                loss = train_results['loss']
            grads = tape.gradient(loss, trained_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, trained_model.trainable_variables))
        #trained_model.save_weights(filepath)
        train_results = trained_model(x_train, y_train, mask)
    
        train_preds = train_results["predictions"]
        #print('\n\n\n Les prédictions sont: \n\n')
        #print(train_preds)
        

        print(f'\n\n The accuracy: {train_results["accuracy"]}')

    trained_model.save_weights("weigths")

    representations = train_results["states"][:5]
    print('\n\n\n Les étatss sont: \n\n')
    print(representations)