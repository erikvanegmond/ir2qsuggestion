from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
from datetime import datetime
import pickle

from RNNTensors.jaimy_model import HRED
import features.adj as adj
from sessionizer import Sessionizer
import utils

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 100
CHECKPOINT_FREQ_DEFAULT = 1000
PRINT_FREQ_DEFAULT = 10
Q_DIM_DEFAULT = 1000
S_DIM_DEFAULT = 1500
VOCAB_DIM_DEFAULT = 90004
NUM_LAYERS_DEFAULT = 1
PADDING_DEFAULT = 50
CLICK_LEVEL = 5
# Directory for tensorflow logs
LOG_DIR_DEFAULT = '../logs'
CHECKPOINT_DIR_DEFAULT = '../checkpoints'
### --- END default constants---

def train():
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)
    
    snizer = Sessionizer()
    train_sess = snizer.get_sessions_with_numbers()
    train_sess_clicks = snizer.get_sessions_clickBool_clickRank()
    snizer = Sessionizer('../data/val_session')
    val_sess = snizer.get_sessions_with_numbers()
    # Choose a random subset to validate on, because otherwise the validation takes too long
    val_sess = np.random.choice(val_sess, 10)
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, 300, FLAGS.num_layers)
    print('[Model was created.]')
    # Feeds for inputs.
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [FLAGS.padding,])
        dec_input = tf.placeholder(tf.int32, [FLAGS.padding,])
        target = tf.placeholder(tf.int32, [FLAGS.padding,])
        s0 = tf.placeholder(tf.float32, [1, FLAGS.s_dim])
        click_hot = tf.placeholder(tf.int32, shape=(1,FLAGS.click_level))
        #click_rank = tf.placeholder(tf.int32, shape=(1, 1))#tf.placeholder(tf.int32, shape=(), name="init")#tf.placeholder(tf.int32, shape=(1, FLAGS.click_level))
    print (click_hot)
    # Data pipeline
    logits, S = model.inference(query, dec_input, s0, click_hot)
    loss = model.loss(logits, target)
    acc = model.accuracy(logits, target)
    # Initialize optimizer
    opt = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    opt_operation = opt.minimize(loss)
    # Create a saver.
    saver = tf.train.Saver()
#    
    with tf.Session() as sess:
        # Initialize summary writers
        merged = tf.merge_all_summaries()
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        train_writer = []#tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = []#tf.summary.FileWriter(FLAGS.log_dir + '/test')
        # Initialize variables
        if FLAGS.resume:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        else:
            sess.run(tf.global_variables_initializer())     
#         Do a loop
        start_time = datetime.now()
        time = start_time.strftime('%d-%m %H:%M:%S')
        print('[%s: Starting training.]' % time)
        num_examples_seen = 0
        for iteration in range(FLAGS.max_steps):
            # Select a random session to train on.
            session = np.random.choice(train_sess)
            idx = train_sess.index(session)
            session_clicks = train_sess_clicks[idx]
            click_ranks = [x[1]-1 if x[1] <=2 else -1 for x in session_clicks]
            state = np.zeros((1,FLAGS.s_dim))
            losses = []
            for i in range(len(session)-1):
                # Loop over the session and predict each query using the previous ones
                
                x1 = pad_query(session[i], pad_size=FLAGS.padding)
                x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
                z = np.zeros([FLAGS.click_level])
                z[click_ranks[i]] = 1
                if i == len(session)-2:
                    # We're at the anchor query of this session
                    _, l, summary = sess.run([opt_operation, loss, merged], feed_dict={query: x1, dec_input: x2, target: y, s0: state, click_hot: z})
                    writer.add_summary(summary, iteration)
                else:
                    _, state, l = sess.run([opt_operation, S, loss], feed_dict={query: x1, dec_input: x2, target: y, s0: state, click_hot: z})
                losses.append(l)
                num_examples_seen += 1
            # Append the loss of this session to the training data
            train_writer.append(np.mean(losses))
            if (iteration+1) % FLAGS.print_freq == 0:
                print('Visited %s examples of %s sessions. Loss: %f' % (num_examples_seen, iteration+1, train_writer[-1]))
            # Evaluate the model
            if iteration % FLAGS.eval_freq == 0 or iteration == FLAGS.max_steps - 1:
                val_losses = []
                # Loop over all the sessions in the validation set
                for session in val_sess:
                    state = np.zeros((1,FLAGS.s_dim))
                    losses = []
                    for i in range(len(session)-1):
                        x1 = pad_query(session[i], pad_size=FLAGS.padding)
                        x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                        y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
                    
                        state, l, accuracy = sess.run([S, loss, acc], feed_dict={query: x1, dec_input: x2, target: y, s0: state, click_hot: z})
                        losses.append(l)
                    val_losses.append(np.mean(losses))
                test_writer.append(np.mean(val_losses))
                print('-' * 40)
                print('Train loss at step %s: %f.' % (iteration, train_writer[-1]))
                print('Test loss at step %s: %f. Accuracy: %f' % (iteration, test_writer[-1], accuracy))
                print('Number of examples seen: %s' % num_examples_seen)
                print('-' * 40)
                break
                # Save the loss data so that we can plot it later
                np.save(FLAGS.log_dir + '/train', np.array(train_writer))
                np.save(FLAGS.log_dir + '/test', np.array(test_writer))
            # Save the model
            if iteration % FLAGS.checkpoint_freq == 0 or iteration == FLAGS.max_steps - 1:
                file_name = FLAGS.checkpoint_dir + '/HRED_model'
                saver.save(sess, file_name, iteration)
                
def feature_extraction():
    """
    This method is used to retrieve the likelohood of different query pairs, given a session.
    """
    # Load data
    ADJ = adj.ADJ()
    start_time = datetime.now()
    time = start_time.strftime('%d-%m %H:%M:%S')
    print("[%s: Loading sessions lm_tr_sessions.pkl]" % time)
    pkl_file = open('../data/lm_tr_sessions.pkl', 'rb')
    sessions = pickle.load(pkl_file)
    pkl_file.close()
    print("[Loaded %s test sessions. It took %f seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, FLAGS.num_layers)
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
        features = {}
        queries = 0
        
        start_time = datetime.now()
        time = start_time.strftime('%d-%m %H:%M:%S')
        print("[%s: Creating features...]" % time)
        for session in sessions:
            # Encode the session
            state = np.zeros((1,FLAGS.s_dim))
            for i in range(len(session)-2):
                # Go through the session step by step to get the session encoding up until the ancor query
                num_query = utils.vectorify(session[i])
                x1 = pad_query(num_query, pad_size=FLAGS.padding)
                num_query = utils.vectorify(session[i+1])
                x2 = pad_query(num_query, pad_size=FLAGS.padding, q_type='dec_input')   
                state, l = sess.run([S], feed_dict={query: x1, dec_input: x2, s0: state})            
            # Get the anchor query
            anchor_query = session[-2]
            adj_dict = ADJ.adj_function(anchor_query)
            highest_adj_queries = adj_dict['adj_queries']
            features[anchor_query] = {}
            # Calculate the likelihood between the queries
            for sug_query in highest_adj_queries:
                num_anchor_query = utils.vectorify(anchor_query)
                x1 = pad_query(num_anchor_query, pad_size=FLAGS.padding)
                num_sug_query = utils.vectorify(sug_query)
                x2 = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(num_sug_query, pad_size=FLAGS.padding, q_type='target')
                # Get the likelihood from the model
                like = sess.run([preds], feed_dict={query: x1, dec_input: x2, s0: state})   
                features[anchor_query][sug_query] = likelihood(like, y)
            queries += 1
            if queries % 10000 == 0:
                print("[Visited %s anchor queries.]" % (queries))
        print("[Saving features %s features.]" % (len(features)))
        pickle.dump(features, open('../data/HRED_features.tf.pkl', 'wb'))
        print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
        
def likelihood(preds, target_query):
    # Calculate the query likelihood without taking the padding into account
    L = 1
    for word in target_query:
        if word == utils.PAD_ID:
            break
        else:
            L *= preds[word]
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
    print(query)
    if len(query) < pad_size:
        if q_type == 'input':
            pad_query = np.array(query + [utils.PAD_ID] * (pad_size - len(query)))
        elif q_type == 'dec_input':
            pad_query = np.array([utils.GO_ID] + query + [utils.PAD_ID] * (pad_size - len(query) - 1))
        elif q_type == 'target':
            pad_query = np.array(query + [utils.EOS_ID] + [utils.PAD_ID] * (pad_size - len(query) - 1))
        else:
            pad_query = None
    else:
        if q_type == 'input':
            pad_query = np.array(query[:pad_size])
        elif q_type == 'dec_input':
            pad_query = np.array([utils.GO_ID] + query[:pad_size-1])
        elif q_type == 'target':
            pad_query = np.array(query[:pad_size-1] + [utils.EOS_ID])
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
    # Print all Flags to confirm parameter settings
    print_flags()
    # Make directories if they do not exists yet
    if not tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    if FLAGS.is_train == 'False':
    # Run feature extraction
        raise NotImplemented
    else:
    # Run the training operation
        train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--q_dim', type = str, default = Q_DIM_DEFAULT,
                        help='Query embedding dimensions')
    parser.add_argument('--s_dim', type = str, default = S_DIM_DEFAULT,
                        help='Session embedding dimensions')
    parser.add_argument('--vocab_dim', type = str, default = VOCAB_DIM_DEFAULT,
                        help='Length of the vocabulary')
    parser.add_argument('--num_layers', type = str, default = NUM_LAYERS_DEFAULT,
                        help='Number of layers in each GRU encoder/decoder')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--padding', type = int, default = PADDING_DEFAULT,
                        help='To what length the queries will be padded.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--resume', type = str, default = False,
                      help='Resume training from latest checkpoint')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    parser.add_argument('--click_level', type = int, default = CLICK_LEVEL,
                        help='Click level for the click feature')
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run()