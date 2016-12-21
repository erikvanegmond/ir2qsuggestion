"""
This script makes the HRED features for the lambdamart data using the tensorflow model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
from datetime import datetime
import pickle
import os
import Pandas as pd

from RNNTensors.TFmodel import HRED
import utils
import lambda_mart as lm

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
DATA_SET_DEFAULT = 'train'
PADDING_DEFAULT = 50
Q_DIM_DEFAULT = 1000
S_DIM_DEFAULT = 1500
VOCAB_DIM_DEFAULT = 90004
NUM_LAYERS_DEFAULT = 1
# Directory for tensorflow logs
CHECKPOINT_DIR_DEFAULT = '../checkpoints/plain_model'
### --- END default constants---
                
def feature_extraction(sessions):
    """
    This method is used to retrieve the likelohood of different query pairs, given a session.
    """
    feature_file = '../data/HRED_noisy_features.tf.pkl'
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, 300, FLAGS.num_layers)
    # Feeds for inputs.
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [FLAGS.padding,])
        dec_input = tf.placeholder(tf.int32, [FLAGS.padding,])
        s0 = tf.placeholder(tf.float32, [1, FLAGS.s_dim])
    # Data pipeline
    logits, S = model.inference(query, dec_input, s0)
    with tf.variable_scope('prediction'):
        preds = tf.nn.softmax(logits)
    # Create a saver adn restore the model.
    saver = tf.train.Saver()
    sess = tf.Session()      
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    print('Model was restored.')
    # Check if you have features already
    if os.path.isfile(feature_file):
        print('[%s already exists, opening file...]')
        features = pickle.load(open(feature_file, 'rb'))
    else:
        features = {}

    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print("[%s: Creating features...]" % time)
    headers = lm.create_dataframe_headers()
    headers.append("HRED")
    queries = 0
    for session in sessions:
        anchor_query = session[-2]
        target_query = session[-1]
        features, highest_adj_queries = lm.create_features(anchor_query, session)
        # Encode the session
        state = np.zeros((1,FLAGS.s_dim))
        for i in range(len(session)-2):
            # Go through the session step by step to get the session encoding up until the ancor query
            num_query = utils.vectorify(session[i])
            x1 = pad_query(num_query, pad_size=FLAGS.padding)
            num_query = utils.vectorify(session[i+1])
            x2 = pad_query(num_query, pad_size=FLAGS.padding, q_type='dec_input')
            state = sess.run(S, feed_dict={query: x1, dec_input: x2, s0: state})
        # Calculate the likelihood between the queries
        if anchor_query not in features.keys():
            print('Unknown anchor query: ' + anchor_query)
            features[anchor_query] = {}
        hred_features = []
        for sug_query in highest_adj_queries:
            if sug_query in features[anchor_query].keys():
                hred_features.append(features[anchor_query][sug_query])
                continue
            else:
                num_anchor_query = utils.vectorify(anchor_query)
                x1 = pad_query(num_anchor_query, pad_size=FLAGS.padding)
                num_sug_query = utils.vectorify(sug_query)
                x2 = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='target')
                # Get the likelihood from the model
                like = sess.run(preds, feed_dict={query: x1, dec_input: x2, s0: state})   
                features[anchor_query][sug_query] = likelihood(like, y)
                hred_features.append(features[anchor_query][sug_query])
        # Add the HRED features to the feature collection
        features = np.vstack((features, np.transpose(np.array(hred_features))))
        target_vector = np.zeros(len(highest_adj_queries))
        [target_query_index] = [q for q, x in enumerate(highest_adj_queries) if x == target_query]
        target_vector[target_query_index] = 1
        sess_data = np.vstack((np.transpose(target_vector), features))
        if queries == 0:
            lambdamart_data = sess_data
            queries += 1
        else:
            lambdamart_data = np.hstack((lambdamart_data, sess_data))
            queries += 1
        if queries % 1000 == 0:
            print("[Visited %s anchor queries.]" % queries)
    lambda_dataframe = pd.DataFrame(data=np.transpose(lambdamart_data), columns=headers)
    lambda_dataframe.to_csv('../data/lamdamart_data_noisy.tf.csv')
    print("---" * 30)
    print("used sessions:" + str(queries))
        
def likelihood(preds, target_query):
    # Calculate the query likelihood without taking the padding into account
    L = 1
    for i, word in enumerate(target_query):
        if word == utils.PAD_ID:
            break
        else:
            L += np.log(preds[i][word])
    return L
    
def pad_query(query, pad_size=50, q_type='input'):
    """
    This method will add padding symbols to make the input query of length pad_size.
    It will also add start/stop symbols if necessary
    Args:
      query: a list of indices representing the query
      pad_size: the size to which the query must be padded
      q_type: a string describing the type of the query. Can be either 'input', 'dec_input' or 'target'
    Returns:
      pad_query: a list of indices representing the padded query
    """
    if len(query) < pad_size:
        if q_type == 'input':
            pad_query = np.append(query, np.array([utils.PAD_ID] * (pad_size - len(query))))
        elif q_type == 'dec_input':
            pad_query = np.append(np.append(np.array([utils.GO_ID]), query), np.array([utils.PAD_ID] * (pad_size - len(query) - 1)))
        elif q_type == 'target':
            pad_query = np.append(np.append(query, np.array([utils.EOS_ID])), np.array([utils.PAD_ID] * (pad_size - len(query) - 1)))
        else:
            pad_query = None
    else:
        if q_type == 'input':
            pad_query = np.array(query[:pad_size])
        elif q_type == 'dec_input':
            pad_query = np.append(np.array([utils.GO_ID]), query[:pad_size-1])
        elif q_type == 'target':
            pad_query = np.append(query[:pad_size-1], np.array([utils.EOS_ID]))
        else:
            pad_query = None
        
    return pad_query

def main(_):
    """
    Main function
    """
    # Load data
    start_time = datetime.now()
    time = start_time.strftime('%d-%m %H:%M:%S')
    data_file = '../data/lm_' + FLAGS.data_set + '_sessions.pkl'
    print("[%s: Loading sessions %s]" % (time, data_file))
    pkl_file = open(data_file, 'rb')
    sessions = pickle.load(pkl_file)
    pkl_file.close()
    print("[Loaded %s test sessions. It took %f seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
    
    # Run feature extraction    
    print('[Creating dataset for noisy_query predictions.]')
    noisy_query_sessions = lm.noisy_query_prediction(sessions)
    feature_extraction(noisy_query_sessions)
    print("---" * 30)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_dim', type = str, default = Q_DIM_DEFAULT,
                        help='Query embedding dimensions')
    parser.add_argument('--s_dim', type = str, default = S_DIM_DEFAULT,
                        help='Session embedding dimensions')
    parser.add_argument('--vocab_dim', type = str, default = VOCAB_DIM_DEFAULT,
                        help='Length of the vocabulary')
    parser.add_argument('--num_layers', type = str, default = NUM_LAYERS_DEFAULT,
                        help='Number of layers in each GRU encoder/decoder')
    parser.add_argument('--padding', type = int, default = PADDING_DEFAULT,
                        help='To what length the queries will be padded.')
    parser.add_argument('--data_set', type = str, default = DATA_SET_DEFAULT,
                        help='Which data set are we going to use? tr, test or val.')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run()