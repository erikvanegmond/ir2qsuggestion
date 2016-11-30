"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

class Decoder():
    
    def __init__(self, q_dim, s_dim, vocab, max_length, bptt_truncate=4, o_dim=300):
        self.params = []
        self.q_dim = q_dim
        self.o_dim = o_dim
        self.s_dim = s_dim
        self.vocab = vocab
        self.max_length = max_length
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        invSqr1 = np.sqrt(1./len(vocab))
        invSqr2 = np.sqrt(1./q_dim)
        D0 = np.random.uniform(-invSqr2, invSqr2, (q_dim, s_dim))
        b0 = np.zeros((q_dim, 1))
        invSqr3 = np.sqrt(1./o_dim)
        Ho = np.random.uniform(-invSqr3, invSqr3, (o_dim, q_dim))
        Eo = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        bo = np.zeros((o_dim, 1))
        O = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        U = np.random.uniform(-invSqr1, invSqr1, (3, q_dim, len(vocab)))
        W = np.random.uniform(-invSqr2, invSqr2, (3, q_dim, q_dim))
        b = np.zeros((q_dim, 3))
        # Theano: Created shared variables
        self.D0 = self.add_to_params(theano.shared(name='decoder/D0', value=D0.astype(theano.config.floatX)))
        self.b0 = self.add_to_params(theano.shared(name='decoder/b0', value=b0.astype(theano.config.floatX)))
        self.Ho = self.add_to_params(theano.shared(name='decoder/Ho', value=Ho.astype(theano.config.floatX)))
        self.Eo = self.add_to_params(theano.shared(name='decoder/Eo', value=Eo.astype(theano.config.floatX)))
        self.bo = self.add_to_params(theano.shared(name='decoder/bo', value=bo.astype(theano.config.floatX)))
        self.O = self.add_to_params(theano.shared(name='decoder/O', value=O.astype(theano.config.floatX)))
        self.U = self.add_to_params(theano.shared(name='decoder/U', value=U.astype(theano.config.floatX)))
        self.W = self.add_to_params(theano.shared(name='decoder/W', value=W.astype(theano.config.floatX)))
        self.b = self.add_to_params(theano.shared(name='decoder/b', value=b.astype(theano.config.floatX)))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
            
    def __theano_build__(self):
        U, W, b = self.U, self.W, self.b
        Ho, Eo, bo, O = self.Ho, self.Eo, self.bo, self.O
        
        def forward_prop_step(w_t_prev, d_t_prev):
            
            # Create the activation of the current word
            omega = Ho.dot(d_t_prev) + Eo.dot(w_t_prev) + bo
            w_t = T.nnet.softmax((O.T).dot(omega)) # Should return a vector of the length of the vocabulary with probs for each word
            print w_t.shape.eval()
            
            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(w_t) + W[0].dot(d_t_prev) + b[:,0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(w_t) + W[1].dot(d_t_prev) + b[:,1])
            c_t = T.tanh(U[2].dot(w_t) + W[2].dot(d_t_prev * r_t) + b[:,2])
            d_t = (T.ones_like(z_t) - z_t) * c_t + z_t * d_t_prev

            return [w_t, d_t]

        s = T.vector('decoder/s')

        h_0 = T.tanh(self.D0.dot(s) + self.b0) # Initialize the first recurrent activation with the session
        w_0 = np.zeros((len(self.vocab), 1)).astype(theano.config.floatX) # The length of the vocab, with 0 probability for every word
        [W, H], updates = theano.scan(forward_prop_step, n_steps=self.max_length, truncate_gradient=self.bptt_truncate,
                                    outputs_info=[dict(initial=w_0), dict(initial=h_0)])
        
        self.forward = theano.function(inputs=[s], outputs=W)
        
    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param