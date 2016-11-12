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
        # Create a GRU to calculate the decoder recurrent state
        self.GRU_dec = GRU(s_dim, q_dim, self.rng)
        # Create output parameters
        D_0 = NormalInit(self.rng, q_dim, s_dim)
        b_0 = np.zeros((q_dim,), dtype='float32')
        W_out = NormalInit(self.rng, q_dim, o_dim)
        Ho = NormalInit(self.rng, o_dim, q_dim)
        Eo = NormalInit(self.rng, o_dim, vocab.shape[1])
        b_out = np.zeros((self.idim,), dtype='float32')
        # Theano: Created shared variables
        self.D_0 = self.add_to_params(theano.shared(name='D_0', value=D_0.astype(theano.config.floatX)))
        self.b_0 = self.add_to_params(theano.shared(name='b_0', value=b_0.astype(theano.config.floatX)))
        self.W_out = self.add_to_params(theano.shared(name='W_out', value=W_out.astype(theano.config.floatX)))
        self.Ho = self.add_to_params(theano.shared(name='Ho', value=Ho.astype(theano.config.floatX)))
        self.Eo = self.add_to_params(theano.shared(name='Eo', value=Eo.astype(theano.config.floatX)))
        self.b_out = self.add_to_params(theano.shared(name='b_out', value=b_out.astype(theano.config.floatX)))
        
    def forward(self, S):
        d_0 = T.tanh(T.dot(self.D_0, S[len(S) - 1]) + self.b_0)
        w_0 = np.zeros(self.vocab.shape[0])
        D, updates = theano.scan(self.GRU_dec.forward_prop_step, sequences=X, outputs_info=[None, d_0])
        
        pre_activ = T.dot(D, self.W_out)
        probs = SoftMax(T.dot(pre_activ, self.W_emb.T) + self.b_out)
        
    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param