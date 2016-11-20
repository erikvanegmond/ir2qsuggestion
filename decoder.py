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
        invSqr1 = np.sqrt(1./len(vocab))
        invSqr2 = np.sqrt(1./q_dim)
        D0 = np.random.uniform(-invSqr2, invSqr2, (q_dim, s_dim))
        b0 = np.zeros((1, s_dim))
        invSqr3 = np.sqrt(1./o_dim)
        Ho = np.random.uniform(-invSqr3, invSqr3, (o_dim, q_dim))
        Eo = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        bo = np.zeros((1,o_dim))
        O = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        U = np.random.uniform(-invSqr1, invSqr1, (3, q_dim, len(vocab)))
        W = np.random.uniform(-invSqr2, invSqr2, (3, q_dim, q_dim))
        b = np.zeros((3, q_dim))
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

            return [w_t, d_t]
        
    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param