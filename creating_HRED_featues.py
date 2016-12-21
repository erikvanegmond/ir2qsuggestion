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
CHECKPOINT_DIR_DEFAULT = '../checkpoints/sessionwise'
### --- END default constants---
                
def feature_extraction(sessions):
    """
    This method is used to retrieve the likelohood of different query pairs, given a session.
    """
    noisy_query_sessions = lm.noisy_query_prediction(sessions)    
    
#    next_query_file = '../data/HRED_features.next_query.tf.pkl'
#    noisy_file = '../data/HRED_features.noisy.tf.pkl'
#    long_tail_file = '../data/HRED_features.long_tail.tf.pkl'
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
    # Create a saver.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:        
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        print('Model was restored.')
        features = {'next_query':{}, 'noisy':{}, 'long_tail':{}}
        queries = 0
        
        start_time = datetime.now()
        time = start_time.strftime('%d-%m %H:%M:%S')
        print("[%s: Creating features...]" % time)
        for s in range(len(sessions)):
            session = sessions[s]
            noisy_session = noisy_query_sessions[s]
            experiments = {'next_query':(), 'noisy':(), 'long_tail':()}
            # Get the anchor queries
            anchor_query = session[-2]
            n_anchor_query = noisy_session[-2]
            lt_anchor_query = session[-2]
            # Iteratively shorten anchor query by dropping terms
            # until we have a query that appears in the Background data
            shortened = False
            for j in range(len(lt_anchor_query.split())):
                [background_count] = lm.bgc.calculate_feature(None, [lt_anchor_query])            
                if background_count == 0 and len(lt_anchor_query.split()) > 1:
                    lt_anchor_query = lm.shorten_query(lt_anchor_query)
                    shortened = True
                else:
                    break                
            # Encode the session
            state = np.zeros((1,FLAGS.s_dim))
            for i in range(len(session)-2):
                # Go through the session step by step to get the session encoding up until the ancor query
                num_query = utils.vectorify(session[i])
                x1 = pad_query(num_query, pad_size=FLAGS.padding)
                # For the long_tail experiment we need to change the anchor query
                if i == len(session)-3 and shortened:
                    num_query = utils.vectorify(lt_anchor_query)
                    x2 = pad_query(num_query, pad_size=FLAGS.padding, q_type='dec_input')
                    lt_state = sess.run(S, feed_dict={query: x1, dec_input: x2, s0: state})
                    experiments['long_tail'] = (lt_anchor_query, lt_state)
                num_query = utils.vectorify(session[i+1])
                x2 = pad_query(num_query, pad_size=FLAGS.padding, q_type='dec_input')
                state = sess.run(S, feed_dict={query: x1, dec_input: x2, s0: state})
                experiments['next_query'] = (anchor_query, state)
                
            # Encode the noisy session
            state = np.zeros((1,FLAGS.s_dim))
            for i in range(len(noisy_session)-2):
                # Go through the session step by step to get the session encoding up until the ancor query
                num_query = utils.vectorify(noisy_session[i])
                x1 = pad_query(num_query, pad_size=FLAGS.padding)
                num_query = utils.vectorify(noisy_session[i+1])
                x2 = pad_query(num_query, pad_size=FLAGS.padding, q_type='dec_input')
                state = sess.run(S, feed_dict={query: x1, dec_input: x2, s0: state})
                experiments['noisy'] = (n_anchor_query, state)
            
            for exp in ['next_query', 'noisy', 'long_tail']:
                # If the query was not shortened, the anchor query (and thus the suggestions) are the same as for next_query
                if exp == 'long_tail' and not shortened:
                    anchor_query, state = experiments['next_query']
                    features[exp][anchor_query] = features['next_query'][anchor_query]
                    continue
                anchor_query, state = experiments[exp]
                adj_dict = lm.adj.adj_function(anchor_query)
                highest_adj_queries = adj_dict['adj_queries']
                if anchor_query not in features[exp].keys():
                    print(exp + ': Unknown anchor query: ' + anchor_query)
                    features[exp][anchor_query] = {}
                # Calculate the likelihood between the queries
                for sug_query in highest_adj_queries:
                    if sug_query in features[exp][anchor_query].keys():
                        continue
                    else:
                        num_anchor_query = utils.vectorify(anchor_query)
                        x1 = pad_query(num_anchor_query, pad_size=FLAGS.padding)
                        num_sug_query = utils.vectorify(sug_query)
                        x2 = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='dec_input')
                        y = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='target')
                        # Get the likelihood from the model
                        like = sess.run(preds, feed_dict={query: x1, dec_input: x2, s0: state})   
                        features[exp][anchor_query][sug_query] = likelihood(like, y)
            queries += 1
                    
            if queries % 10000 == 0:
                print("[Visited %s anchor queries. It took %d seconds.]" % (queries, (datetime.now() - start_time).seconds))
        print("[Saving features %s features.]" % (queries))
        for exp, f in features:
            pickle.dump(f, open('../data/HRED_features.'+exp+'.tf.pkl', 'wb'))
        print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
        
def likelihood(preds, target_query):
    # Calculate the query likelihood without taking the padding into account
    L = 1
    for i, word in enumerate(target_query):
        if word == utils.PAD_ID:
            break
        else:
            L *= preds[i][word]
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
    
def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
      print(key + ' : ' + str(value))

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
    feature_extraction(sessions)
#    print('[Creating dataset for next_query predictions.]')
#    feature_extraction(sessions, 'next_query')
#    print("---" * 30)
#    
#    print('[Creating dataset for noisy_query predictions.]')
#    noisy_query_sessions = lm.noisy_query_prediction(sessions)
#    feature_extraction(noisy_query_sessions, 'noisy')
#    print("---" * 30)
#    
#    print('[Creating dataset for long_tail_query predictions.]')
#    feature_extraction(sessions, 'long_tail', True)
#    print("---" * 30)

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