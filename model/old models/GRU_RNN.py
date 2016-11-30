"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

# This class can be used for any en- or decode step.
class Query_GRU():
    
    def __init__(self, in_dim, out_dim, emb_dim=300, bptt_truncate=4, scope=''):
        # The parameters
        self.params = []
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bptt_truncate = bptt_truncate
        self.scope = scope
        
        # Initialize the network parameters
        invSqr1 = np.sqrt(1./in_dim)
        invSqr2 = np.sqrt(1./emb_dim)
        invSqr3 = np.sqrt(1./out_dim)
        E = np.random.uniform(-invSqr1, invSqr1, (emb_dim, in_dim))
        U = np.random.uniform(-invSqr2, invSqr2, (3, out_dim, emb_dim))
        W = np.random.uniform(-invSqr3, invSqr3, (3, out_dim, out_dim))
        b = np.zeros((3, out_dim))
        # Theano: Created shared variables
        self.E = self.add_to_params(theano.shared(name=scope+'/E', value=E.astype(theano.config.floatX)))
        self.U = self.add_to_params(theano.shared(name=scope+'/U', value=U.astype(theano.config.floatX)))
        self.W = self.add_to_params(theano.shared(name=scope+'/W', value=W.astype(theano.config.floatX)))
        self.b = self.add_to_params(theano.shared(name=scope+'/b', value=b.astype(theano.config.floatX)))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        E, U, W, b = self.E, self.U, self.W, self.b
        
        def forward_prop_step(x_t, h_t_prev):
            x_emb = E[:,x_t]
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_emb) + W[0].dot(h_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_emb) + W[1].dot(h_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_emb) + W[2].dot(h_t_prev * r_t) + b[2])
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_t_prev
    
            return [h_t]
        
        x = T.ivector(self.scope+'/x')
        h_0 = np.zeros(self.out_dim).astype(theano.config.floatX)
        
        H, updates = theano.scan(forward_prop_step, sequences=x, truncate_gradient=self.bptt_truncate,
                                          outputs_info=[h_0])

        self.forward = theano.function([x], H)

    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param
        
# This class can be used for any en- or decode step.
class Session_GRU():
    
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
        
        def forward_prop_step(x_t, h_t_prev):

            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_t) + W[0].dot(h_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_t) + W[1].dot(h_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_t) + W[2].dot(h_t_prev * r_t) + b[2])
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_t_prev
    
            return [h_t]
        
        x = T.vector(self.scope+'/x')
        h_0 = T.vector(self.scope+'/h_0')
        
        H = forward_prop_step(x, h_0)

        self.forward = theano.function(inputs=[x, h_0], outputs=H)

    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param