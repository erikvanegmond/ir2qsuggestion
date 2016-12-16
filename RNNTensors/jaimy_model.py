"""
This module implements a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

import utils

class HRED(object):
  """
  This class implements a Multilayer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference and evaluation.
  """

  def __init__(self, vocab_size, q_dim, s_dim, o_dim, num_layers, num_samples=512, is_training=True, click_level=5):
    """
    Constructor for an HRED object.
    Args:
      vocab_size:   The size of the vocabulary
      q_dim:        The size of the query embeddings
      s_dim:        The size of the session embeddings
      o_dim:        The size of the output
    """
    self.vocab_size = vocab_size
    self.q_dim = q_dim
    self.s_dim = s_dim
    self.o_dim = o_dim
    self.num_layers = num_layers
    self.num_samples = num_samples
    self.is_training = is_training
    self.click_level = click_level
    
    self.init = initializers.xavier_initializer()
    self.reg = regularizers.l2_regularizer(1e-2)
    self.counter = 0

  def inference(self, query, target, sess_enc, click_hot):
    """
    Given a session x, this method will encode each query in x. After that it
    will encode the session given the query states. Eventually a new query will
    be decoded and returned.
    Args:
      query: a tensor of shape [num_of_query_words, 1] representing the query we need to encode
      target: a tensor of shape [num_of_target_words, 1] representing the target we need to decode to
      sess_enc: a tensor of shape [s_dim, 1] representing the previous session encoding
    Returns:
      logits: a tensor of shape [num_of_target_words, q_dim] representing the decoded query
    """
    with tf.variable_scope('HRED'):        
        E = tf.get_variable('embedding', (self.vocab_size, self.q_dim), initializer=self.init, regularizer=self.reg)
        click_hot = tf.get_variable('hot', (1, self.click_level), initializer=self.init)
        with tf.variable_scope('QueryEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # Loop over all the queries to encode them
            word_embeddings = tf.nn.embedding_lookup(E, query)
            word_embeddings = tf.split(0, word_embeddings.get_shape()[0].value, word_embeddings)#unpack_sequence(word_embeddings)
            _, Q = tf.nn.rnn(cell, word_embeddings, dtype=tf.float32)
            Q = tf.concat(1, [Q, click_hot])
        with tf.variable_scope('SessionEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.s_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # When we're not training we only want to predict the last query
            _, S = cell(Q, sess_enc)
        with tf.variable_scope('Decoder') as dec:
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # Get/create the variables
            H0 = tf.get_variable('weights', (self.s_dim, self.q_dim), initializer=self.init, regularizer=self.reg)
            b0 = tf.get_variable('bias', (1, self.q_dim), initializer=tf.constant_initializer(0.0))
            Ho = tf.get_variable('omega_Hweights', (self.q_dim, self.o_dim), initializer=self.init, regularizer=self.reg)
            Eo = tf.get_variable('omega_Eweights', (self.q_dim, self.o_dim), initializer=self.init, regularizer=self.reg)
            bo = tf.get_variable('omega_bias', (1, self.o_dim), initializer=tf.constant_initializer(0.0))
            O = tf.get_variable('embedding', (self.o_dim, self.vocab_size), initializer=self.init, regularizer=self.reg)
            # According to the paper, this is how s is used to generate the query
            state = tf.tanh(tf.matmul(S, H0) + b0)
            word_embeddings = tf.nn.embedding_lookup(E, target)
            word_embeddings = tf.split(0, word_embeddings.get_shape()[0].value, word_embeddings)
            D = []
            for word in word_embeddings:    
                D.append(state)                
                _, state = cell(word, state)          
                dec.reuse_variables()
            # We add a final linear layer to create the output                        
            D = pack_sequence(D)
            W = pack_sequence(word_embeddings)
            omega = tf.matmul(D, Ho) + tf.matmul(W, Eo) + bo
            
            logits = tf.matmul(omega, O)

    return logits, S

  def loss(self, logits, labels):
    """
    Calculates the sofmax loss using sampling. 
    Args:
      logits: A list of 2D float Tensor of size [num_words, self.vocab_size].
      labels: A list of 2D int Tensor of size [num_words, 1] containing the true words of the sequence
    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.variable_scope('loss'):
        lbls = tf.one_hot(labels, self.vocab_size, axis=-1)
        inpts = tf.cast(logits, tf.float32)
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(inpts, lbls)
        softmax_loss = tf.reduce_mean(softmax_loss)
        tf.summary.scalar('softmax loss', softmax_loss)
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_losses)
        tf.summary.scalar('regularization loss', reg_loss)
        
        loss = tf.add(softmax_loss, reg_loss)
        tf.summary.scalar('total loss', loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return loss
    
  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [num_words, self.vocab_size].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [num_words, 1]
                 with one-hot encoding. Ground truth labels for
                 each observation in batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    with tf.name_scope('accuracy'):
      preds = tf.nn.softmax(logits)
      correct_prediction = tf.cast(tf.equal(tf.cast(tf.argmax(preds,1), tf.int32), labels), tf.float32)
      query_len = tf.reduce_sum(tf.cast(tf.not_equal(labels, utils.PAD_ID), tf.int32))
      accuracy = tf.reduce_mean(correct_prediction[:query_len])/tf.cast(query_len, tf.float32)
      tf.summary.scalar('accuracy', accuracy)
    ########################
    # END OF YOUR CODE    #
    ########################

    return accuracy
    
def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    try:
        tensor = tf.reshape(tensor, (tensor.get_shape()[0].value, tensor.get_shape()[2].value))
    except ValueError:
        print(tensor.get_shape())
        raise
    return tf.split(0, tensor.get_shape()[0].value, tensor)

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    dim = sequence[0].get_shape()[1].value
    return tf.reshape(tf.pack(sequence), (len(sequence), dim))