"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from utils import * # NormalInit, OrthogonalInit

# This class can be used for any en- or decode step.
class GRU():
    
    def __init__(self, in_dim, out_dim, rng=None):
        # The parameters
        self.params = []
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Random number generator
        self.rng = rng
        if self.rng == None:
            self.rng = np.random
            self.rng.seed(1234)
        # Initialize the network parameters
        U = OrthogonalInit(self.rng, (3, in_dim, out_dim))
        W = OrthogonalInit(self.rng, (3, out_dim, out_dim))
        b = np.zeros((3, out_dim))
        # Theano: Created shared variables
        self.U = self.add_to_params(theano.shared(name='U', value=U.astype(theano.config.floatX)))
        self.W = self.add_to_params(theano.shared(name='W', value=W.astype(theano.config.floatX)))
        self.b = self.add_to_params(theano.shared(name='b', value=b.astype(theano.config.floatX)))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        U, W, b = self.U, self.W, self.b
        
        def forward_prop_step(x_t, s_t_prev):
            
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(s_t_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev

            return s_t

    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param