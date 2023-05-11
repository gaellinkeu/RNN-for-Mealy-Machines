import tensorflow as tf
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Embedding
from keras.losses import CategoricalCrossentropy

class Tagger(tf.keras.Model):

  def __init__(self, n_tokens = 3, embed_dim = 10, max_length = 99, rnn_dim = 10, n_labels=3):
    super().__init__()
    self.embedding = Embedding(n_tokens, embed_dim, input_length=max_length, mask_zero=True)
    self.rnn = SimpleRNN(rnn_dim, return_sequences=True)
    self.outputs = Dense(n_labels, activation='softmax')

  def call(self, token_ids, labels, mask, training = True):
    embeddings = self.embedding(token_ids)
    states = self.rnn(embeddings)
    logits = self.outputs(states)
    loss = CategoricalCrossentropy()(labels, logits)
    #predictions = tf.math.argmax(logits, axis=-1)
    bool_acc = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(bool_acc, tf.float32))
    #acc = ((predictions == labels) * mask).sum().float() / mask.sum()
    return {
            "states": states,
            "predictions": logits,
            "accuracy": accuracy.numpy(),
            "loss": loss,
          }


