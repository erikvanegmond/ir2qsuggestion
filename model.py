"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from GRU_RNN import GRU

class Model():
    
    def __init__(self):
        # Random number generator
        self.rng = np.random.RandomState(1234)
        # Create a query encoder
        """
        You can see this as the Encoder class without the session weights (Ws)
        """
        
        # Create session encoder
        """
        You can see this as the Encoder class without the query weights (W)
        """
        self.session_encoder = GRU(1000, 1500, self.rng)
        # Create decoder
        """
        This is the same as the Decoder class
        """
    
    # This method creates session encodings given data x
    def forward(self, x):
        # Encode all the queries in x
        """
        Should look like something in build_encoder (lines 165-167)
        """
        Q = []
        # Encode all the session with the given query encodings
        s_0 = T.alloc(np.float32(0), x.shape[1], self.session_encoder.h_dim)
        S, updates = theano.scan(self.session_encoder.forward_prop_step, sequences=Q, outputs_info=[None, s_0])       
        # Decode the given session encodings
        """
        Should look like something in build_decoder
        """
        
        return x
        
    def backward(self, probs, x_data, x_max_length, y_rank, y_rank_mask, training_x_cost_mask):
        # Create a ranking from the probabilities
        pred_ranks = []
        # Caluclate the cost
        per_example_cost = -T.log2(probs).reshape((x_max_length, x_data.shape[1]))
        rank_cost = T.sum(((pred_ranks[1:].flatten() - y_rank) ** 2) * (y_rank_mask)) / T.sum(y_rank_mask)
        training_cost = T.sum(-T.log2(probs) * training_x_cost_mask) + np.float32(self.lambda_rank) * rank_cost
        
        # Caluclate the gradients
        """
        Should look like something in compute_updates (lines 424-452)
        """        
        # Update the encoders and decoders
        """
        Should look like something in build_train_function (lines 454-460)
        """

    def fit(self):
        # We could do something like SGD or something.
        return False