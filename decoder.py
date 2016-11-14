"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from utils import *
from GRU_RNN import GRU

class Decoder():
    
    def __init__(self, q_dim, s_dim, vocab, o_dim=300, rng=None):
        self.params = []
        self.q_dim = q_dim
        self.o_dim = o_dim
        self.s_dim = s_dim
        self.vocab = vocab
        # Random number generator
        self.rng = rng
        if self.rng == None:
            self.rng = np.random.RandomState(1234)
        # Initialize the network parameters
        D0 = NormalInit(self.rng, (q_dim, s_dim))
        b0 = np.zeros((1,s_dim))
        Ho = NormalInit(self.rng, (o_dim, q_dim))
        Eo = NormalInit(self.rng, (o_dim, vocab.shape[0]))
        bo = np.zeros((1,o_dim))
        O = OrthogonalInit(self.rng, (vocab.shape[1], o_dim))
        U = OrthogonalInit(self.rng, (3, vocab.shape[1], q_dim))
        W = OrthogonalInit(self.rng, (3, q_dim, q_dim))
        b = np.zeros((3, out_dim))
        # Theano: Created shared variables
        self.D0 = self.add_to_params(theano.shared(name='D0', value=D0.astype(theano.config.floatX)))
        self.b0 = self.add_to_params(theano.shared(name='b0', value=b0.astype(theano.config.floatX)))
        self.Ho = self.add_to_params(theano.shared(name='Ho', value=Ho.astype(theano.config.floatX)))
        self.Eo = self.add_to_params(theano.shared(name='Eo', value=Eo.astype(theano.config.floatX)))
        self.bo = self.add_to_params(theano.shared(name='bo', value=bo.astype(theano.config.floatX)))
        self.O = self.add_to_params(theano.shared(name='O', value=O.astype(theano.config.floatX)))
        self.U = self.add_to_params(theano.shared(name='U', value=U.astype(theano.config.floatX)))
        self.W = self.add_to_params(theano.shared(name='W', value=W.astype(theano.config.floatX)))
        self.b = self.add_to_params(theano.shared(name='b', value=b.astype(theano.config.floatX)))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        U, W, b = self.U, self.W, self.b
        Ho, Eo, bo, O = self.Ho, self.Eo, self.bo, self.O
        
        def forward_prop_step(q_t, w_t_prev, d_t_prev):
            
            # Create the activation of the current word
            omega = Ho.dot(d_t_prev) + Eo.dot(w_t_prev) + bo
            w_t = T.nnet.softmax((O.T).dot(omega)) # Should return a vector of the length of the vocabulary with probs for each word
            
            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(w_t) + W[0].dot(d_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(w_t) + W[1].dot(d_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(w_t) + W[2].dot(d_t_prev * r_t) + b[2])
            d_t = (T.ones_like(z_t) - z_t) * c_t + z_t * d_t_prev

            return w_t, d_t
            
    def forward(self, s, q_m):
        h_0 = T.tanh(self.D0.dot(s) + self.b0) # Initialize the first recurrent activation with the session
        w_0 = np.zeros(self.vocab.shape[1]) # The length of the vocab, with 0 probability for every word
        D, W, updates = theano.scan(self.forward_prop_step, sequences=q_m, outputs_info=[w_0, h_0])
        
    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param