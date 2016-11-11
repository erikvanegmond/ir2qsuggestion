"""
@author: Jaimy
"""
import numpy as np
import theano.tensor as T

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
        # Encode all the session with the given query encodings
        """
        Should look like something in build_encoder (lines 176-178)
        """        
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