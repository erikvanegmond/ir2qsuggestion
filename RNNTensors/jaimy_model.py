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

  def __init__(self, vocab_size, q_dim, s_dim, num_layers, num_samples=512, is_training=True):
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
    self.counter = 0

  def inference(self, query, target, sess_enc):
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
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.variable_scope('HRED'):        
        E = tf.get_variable('embedding', (self.vocab_size, self.q_dim), initializer=self.init, regularizer=self.reg)
        with tf.variable_scope('QueryEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # Loop over all the queries to encode them
            word_embeddings = tf.nn.embedding_lookup(E, query)
            word_embeddings = word_embeddings = tf.split(0, word_embeddings.get_shape()[0].value, word_embeddings)#unpack_sequence(word_embeddings)
            _, Q = tf.nn.rnn(cell, word_embeddings, dtype=tf.float32)
        with tf.variable_scope('SessionEncoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.s_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # When we're not training we only want to predict the last query
            _, S = cell(Q, sess_enc)
        with tf.variable_scope('Decoder'):
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            H0 = tf.get_variable('weights', (self.s_dim, self.q_dim), initializer=self.init, regularizer=self.reg)
            b0 = tf.get_variable('bias', (1,self.q_dim), initializer=tf.constant_initializer(0.0))
            # According to the paper, this is how s is used to generate the query
            init_state = tf.matmul(S, H0) + b0
            word_embeddings = tf.nn.embedding_lookup(E, target)
            word_embeddings = word_embeddings = tf.split(0, word_embeddings.get_shape()[0].value, word_embeddings)#unpack_sequence(word_embeddings)
            # query words is a length T list of outputs (one for each input), or a nested tuple of such elements.
            query_words, _ = tf.nn.rnn(cell, word_embeddings, initial_state=init_state)
            logits = pack_sequence(query_words)
    ########################
    # END OF YOUR CODE    #
    #######################

    return logits, S

  def loss(self, logits, labels):
    """
    Calculates the sofmax loss using sampling. 
    Args:
      logits: A list of 2D float Tensor of size [num_words, self.q_dim].
      labels: A list of 2D int Tensor of size [num_words, 1] containing the true words of the sequence
    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.variable_scope('loss'):
        W = tf.get_variable("weights", (self.vocab_size, self.q_dim), dtype=tf.float32, initializer=self.init, regularizer=self.reg)
        b = tf.get_variable("bias", (self.vocab_size,), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        #softmax_losses = []
        #for i in range(len(labels)):
        lbls = tf.reshape(labels, [-1, 1])
        inpts = tf.cast(logits, tf.float32)
        softmax_loss = tf.cast(tf.nn.sampled_softmax_loss(W, b, inpts, lbls,
                                   self.num_samples, self.vocab_size), tf.float32)
        #softmax_losses = tf.reshape(tf.concat(0, softmax_losses), [-1, len(labels)])
        #softmax_loss = tf.reduce_mean(softmax_losses)
        tf.scalar_summary('softmax loss', softmax_loss)
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_losses)
        tf.scalar_summary('regularization loss', reg_loss)
        
        loss = tf.add(softmax_loss, reg_loss)
        tf.scalar_summary('total loss', loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return loss
    
  def session_inference(self, session, targets):
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
        with tf.variable_scope('QueryEncoder') as q_enc:
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            # Loop over all the queries to encode them
            Q = []            
            for query in session:
                word_embeddings = tf.nn.embedding_lookup(E, query)
                word_embeddings = unpack_sequence(word_embeddings)
                _, state = tf.nn.rnn(cell, word_embeddings, dtype=tf.float32)
                Q.append(state)
                q_enc.reuse_variables()
        with tf.variable_scope('SessionEncoder') as s_enc:
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.s_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
                
            if self.is_training:
                # When training we try to predict each query subsequently
                S = []
                for i in range(len(Q)):
                    queries = Q[:i+1]
                    _, s = tf.nn.rnn(cell, queries, dtype=tf.float32)
                    S.append(s)
                    s_enc.reuse_variables()
            else:
                # When we're not training we only want to predict the last query
                _, [S] = tf.nn.rnn(cell, Q)
        with tf.variable_scope('Decoder') as dec:
            # Create the GRU cell(s)
            single_cell = tf.nn.rnn_cell.GRUCell(self.q_dim)
            if self.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
            else:
                cell = single_cell
            H0 = tf.get_variable('weights', (self.s_dim, self.q_dim), initializer=self.init, regularizer=self.reg)
            b0 = tf.get_variable('bias', (1,self.q_dim), initializer=tf.constant_initializer(0.0))
            # We assume that S has the same length as targets
            logits = []
            for i in range(len(S)):
                s = S[i]
                # According to the paper, this is how s is used to generate the query
                init_state = tf.matmul(s, H0) + b0
                word_embeddings = tf.nn.embedding_lookup(E, targets[i])
                word_embeddings = unpack_sequence(word_embeddings)
                # query words is a length T list of outputs (one for each input), or a nested tuple of such elements.
                query_words, _ = tf.nn.rnn(cell, word_embeddings, initial_state=init_state)
                logits.append(pack_sequence(query_words))
                dec.reuse_variables()
    ########################
    # END OF YOUR CODE    #
    #######################

    return logits
    
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