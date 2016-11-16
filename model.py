"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from GRU_RNN import GRU
from decoder import Decoder

class Model():
    
    def __init__(self, vocab, bptt_truncate, max_length):
        # Random number generator
        self.rng = np.random.RandomState(1234)
        self.bptt_truncate = bptt_truncate
        self.max_length = max_length
        # Create a query encoder
        """
        You can see this as the Encoder class without the session weights (Ws)
        """
        self.query_encoder = GRU(len(vocab), 1000, self.rng, 'query')
        # Create session encoder
        """
        You can see this as the Encoder class without the query weights (W)
        """
        self.session_encoder = GRU(1000, 1500, self.rng, 'session')
        # Create decoder
        """
        This is the same as the Decoder class
        """
        self.decoder = Decoder(1000, 1500, vocab, rng=self.rng)
        
        self.theano = {}
        self.__theano_build__()
        
    def __theano_build__(self):
        query_encoder, session_encoder, decoder = self.query_encoder, self.session_encoder, self.decoder
        bptt_truncate, max_length = self.bptt_truncate, self.max_length
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        """
        This method passes the data through the whole process to generate a new query.
        - x_data: contains one or more queries as sequences of one-hot-encoded words
        - s_0: is a vector describing the initial activation of the session encoder (zeros if there were no queries of the same session before this)
        """
        def forward_step(x_data, s_0):
            # Encode all the queries in x
            q_0 = T.zeros(query_encoder.out_dim)
            [Q], updates = theano.scan(query_encoder.forward_prop_step, sequences=x_data, truncate_gradient=bptt_truncate, 
                                     outputs_infor=[None, q_0])
            # Encode all the session with the given query encodings
            [S], updates = theano.scan(session_encoder.forward_prop_step, sequences=Q, truncate_gradient=bptt_truncate,
                                     outputs_info=[None, s_0])       
            # Decode the given session encodings
            h_0 = T.tanh(decoder.D0.dot(S[-1]) + decoder.b0) # Initialize the first recurrent activation with the session
            w_0 = T.zeros(len(decoder.vocab)) # The length of the vocab, with 0 probability for every word
            [W, H], updates = theano.scan(decoder.forward_prop_step, n_steps=max_length, truncate_gradient=bptt_truncate,
                                        outputs_info=[None, w_0, h_0])
            
            return [W, S]
        
        # This can only be done for one session
        s_0 = T.zeros(session_encoder.out_dim)
        [W, S], updates = theano.scan(forward_step, sequences=x, truncate_gradient=bptt_truncate,
                                          outputs_info=[None, s_0])
        prediction = T.argmax(W, axis=1)
        loss = T.sum(T.nnet.categorical_crossentropy(W, y))
        
        params = query_encoder.params + session_encoder.params + decoder.params
        dparams = T.grad(loss, params)
        
        self.predict = theano.function([x], W)
        self.predict_query = theano.function([x], prediction)
        self.cross_error = theano.function([x, y], loss)
        self.bptt = theano.function([x, y], dparams)
        
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        temp = 1 - decay
        cache = decay * self.cache + temp * np.power(dparams, 2)
        
        self.sgd_step = theano.function([x, y, learning_rate, theano.In(decay, value=0.9)], [], 
                                         updates = [(params, np.subtract(params, learning_rate * dparams / T.sqrt(np.add(cache, 1e-6)))),
                                                    (self.cache, cache)])