"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from utils import * # NormalInit, OrthogonalInit

class SessionEncDec():
    
    def __init__(self, q_dim, h_dim=1500, bptt_truncate=4, rng=None):
        # The parameters
        #self.params = []
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.bptt_truncate = bptt_truncate
        self.session_step_type = session_step_type
        # Random number generator
        self.rng = rng
        if self.rng == None:
            self.rng = np.random
            self.rng.seed(1234)
        # Initialize the network parameters
        E = NormalInit(self.rng, h_dim, q_dim)
        U = OrthogonalInit(self.rng, (3, h_dim, h_dim))
        W = OrthogonalInit(self.rng, (3, h_dim, h_dim))
        b = np.zeros((3, hidden_dim))
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        E, U, W, b = self.E, self.U, self.W, self.b
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t1_prev):
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            return s_t1
            
        s, updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, 
                          dict(initial=T.zeros(self.h_dim)),
                          dict(initial=T.zeros(self.h_dim))])