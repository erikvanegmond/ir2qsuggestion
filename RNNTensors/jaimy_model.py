"""
This module implements a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

class HRED(object):
  """
  This class implements a Multilayer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference and evaluation.
  """

  def __init__(self, vocab_size, q_dim, s_dim, num_layers, num_samples, is_training=True):
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
    self.num_layers = num_layers
    self.num_samples = num_samples
    self.is_training = is_training
    
    self.init = initializers.xavier_initializer()
    self.reg = regularizers.l2_regularizer(1e-2)

  def inference(self, session, targets):
    """
    Given a session x, this method will encode each query in x. After that it
    will encode the session given the query states. Eventually a new query will
    be decoded and returned.
    Args:
      session: a list containing queries
      targets: a list containing queries used by the decoder
    Returns:
      logits: a list containing the decoder outputs per target
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.variable_scope('HRED'):        
        E = tf.get_variable('embedding', (self.vocab_size, self.q_dim), initializer=self.init, regularizer=self.reg)
        with tf.variable_scope('QueryEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim, name='single_GRU')
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers, name='mult_GRU')
            else:
                cell = single_cell
            # Loop over all the queries to encode them
            Q = []
            for query in session:
                word_embeddings = tf.nn.embedding_lookup(E, query)
                _, state = tf.nn.rnn(cell, word_embeddings)
                Q.append(state)
        with tf.variable_scope('SessionEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.s_dim, name='single_GRU')
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers, name='mult_GRU')
            else:
                cell = single_cell
                
            if self.is_training:
                # When training we try to predict each query subsequently
                S = []
                for i in range(len(Q)):
                    queries = Q[:i+1]
                    outputs, s = tf.nn.rnn(cell, queries)
                    S.append(s)
            else:
                # When we're not training we only want to predict the last query
                _, [S] = tf.nn.rnn(cell, Q)
        with tf.variable_scope('Decoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.o_dim, name='single_GRU')
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers, name='mult_GRU')
            else:
                cell = single_cell
            H0 = tf.get_variable('weights', (self.q_dim, self.s_dim), initializer=self.init, regularizer=self.reg)
            b0 = tf.get_variable('bias', (self.q_dim), initializer=tf.constant_initializer(0.0))
            # We assume that S has the same length as targets
            logits = []
            for i in range(len(S)):
                s = S[i]
                # According to the paper, this is how s is used to generate the query
                init_state = tf.matmul(H0, s) + b0
                word_embeddings = tf.nn.embedding_lookup(E, targets[i])
                # query words is a length T list of outputs (one for each input), or a nested tuple of such elements.
                query_words, _ = tf.nn.rnn(cell, word_embeddings, initial_state=init_state)
                logits.append[query_words]
    ########################
    # END OF YOUR CODE    #
    #######################

    return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.
    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    You can use tf.scalar_summary to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.
    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   observation in batch.
    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.variable_scope('loss'):
        W = tf.get_variable("weights", (self.vocab_size, self.o_dim), dtype=tf.float32, initializer=self.init, regularizer=self.reg)
        b = tf.get_variable("bias", (self.vocab_size,), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        softmax_loss = []
        for i in range(len(logits)):
            lbls = tf.reshape(labels[i], [-1, 1])
            inpts = tf.cast(logits[i], tf.float32)
            softmax_loss.append(tf.cast(tf.nn.sampled_softmax_loss(W, b, inpts, lbls,
                                           self.num_samples, self.vocab_size), tf.float32))
        softmax_loss = tf.reshape(tf.concat(1, softmax_loss), [-1, len(labels)])
        loss = tf.reduce_mean(softmax_loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return loss