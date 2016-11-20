"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from utils import * # NormalInit, OrthogonalInit

# This class can be used for any en- or decode step.
class GRU():
    
    def __init__(self, in_dim, out_dim, bptt_truncate=4, scope=''):
        # The parameters
        self.params = []
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bptt_truncate = bptt_truncate
        self.scope = scope
        
        # Initialize the network parameters
        invSqr1 = np.sqrt(1./in_dim)
        invSqr2 = np.sqrt(1./out_dim)
        U = np.random.uniform(-invSqr1, invSqr1, (3, out_dim, in_dim))
        W = np.random.uniform(-invSqr2, invSqr2, (3, out_dim, out_dim))
        b = np.zeros((3, out_dim))
        # Theano: Created shared variables
        self.U = self.add_to_params(theano.shared(name=scope+'/U', value=U.astype(theano.config.floatX)))
        self.W = self.add_to_params(theano.shared(name=scope+'/W', value=W.astype(theano.config.floatX)))
        self.b = self.add_to_params(theano.shared(name=scope+'/b', value=b.astype(theano.config.floatX)))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        U, W, b = self.U, self.W, self.b
        
        x = T.vector(self.scope+'x')
        h_0 = T.vector(self.scope+'h_0')
        
        def forward_prop_step(x_t, h_t_prev):
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(h_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(h_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(h_t_prev * r_t) + b[2])
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_t_prev
    
            return [h_t.astype(theano.config.floatX)]

        H, updates = theano.scan(forward_prop_step, sequences=x, truncate_gradient=self.bptt_truncate,
                                          outputs_info=[h_0])

        self.forward = theano.function([x, h_0], H)

    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param