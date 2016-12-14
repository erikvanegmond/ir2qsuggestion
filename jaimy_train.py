from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
import pickle as pkl
from datetime import datetime

from RNNTensors.jaimy_model import HRED
from sessionizer import Sessionizer
import utils

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
Q_DIM_DEFAULT = 1000
S_DIM_DEFAULT = 1500
VOCAB_DIM_DEFAULT = 90004
NUM_LAYERS_DEFAULT = 1
PADDING_DEFAULT = 50
# Directory for tensorflow logs
LOG_DIR_DEFAULT = '../logs'
CHECKPOINT_DIR_DEFAULT = '../checkpoints'
### --- END default constants---

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [(5, 10), (10, 15), (25, 30), (100, 100)]
            
def train_step(losses, params, learning_rate, max_gradient_norm, global_step):
    gradient_norms = []
    updates = []
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    for b in xrange(len(_buckets)):
        gradients = tf.gradients(losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        gradient_norms.append(norm)
        updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step))
        
def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex

def train():
    snizer = Sessionizer()
    train_sess = snizer.get_sessions_with_numbers()
    snizer = Sessionizer('../data/val_session')
    val_sess = snizer.get_sessions_with_numbers()
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, FLAGS.num_layers)
    print('[Model was created.]')
    # Feeds for inputs.
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [FLAGS.padding,])
        dec_input = tf.placeholder(tf.int32, [FLAGS.padding,])
        target = tf.placeholder(tf.int32, [FLAGS.padding,])
        s0 = tf.placeholder(tf.float32, [1, FLAGS.s_dim])
    # Data pipeline
    logits, S = model.inference(query, dec_input, s0)
    loss = model.loss(logits, target)
    # Initialize optimizer
    opt = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    opt_operation = opt.minimize(loss)
    # Create a saver.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Initialize summary writers
#        merged = tf.merge_all_summaries()
        train_writer = []#tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = []#tf.summary.FileWriter(FLAGS.log_dir + '/test')
        # Initialize variables
        sess.run(tf.global_variables_initializer())     
        # Do a loop
        start_time = datetime.now()
        time = start_time.strftime('%d-%m %H:%M:%S')
        print('[%s: Starting training.]' % time)
        num_examples_seen = 0
        for iteration in range(FLAGS.max_steps):
            # Select a random session to train on.
            session = np.random.choice(train_sess)
            state = np.zeros((1,FLAGS.s_dim))
            losses = []
            for i in range(len(session)-1):
                # Loop over the session and predict each query using the previous ones
                x1 = pad_query(session[i], pad_size=FLAGS.padding)
                x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
            
                _, state, l = sess.run([opt_operation, S, loss], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                losses.append(l)
                num_examples_seen += 1
            # Append the loss of this session to the training data
            train_writer.append(np.mean(losses))
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
                    
                        state, l = sess.run([S, loss], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                        losses.append(l)
                    val_losses.append(np.mean(losses))
                test_writer.append(np.mean(val_losses))
                print('Train loss at step %s: %f.' % (iteration, train_writer[-1]))
                print('Test loss at step %s: %f.' % (iteration, test_writer[-1]))
                print('Number of examples seen: %s' % num_examples_seen)
                print('-' * 40)
                # Save the loss data so that we can plot it later
                np.save(open(FLAGS.log_dir + '/train.pnz', 'w'), np.array(train_writer))
                np.save(open(FLAGS.log_dir + '/test.pnz', 'w'), np.array(test_writer))
            # Save the model
            if iteration % FLAGS.checkpoint_freq == 0 or iteration == FLAGS.max_steps - 1:
                file_name = FLAGS.checkpoint_dir + '/HRED_model.ckpt'
                saver.save(sess, file_name)
    
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
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run()