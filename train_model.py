from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
from datetime import datetime

from RNNTensors.TFmodel import HRED
from sessionizer import Sessionizer
import utils

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 0.1#2e-3
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
LOG_DIR_DEFAULT = '../logs/sessionwise'
CHECKPOINT_DIR_DEFAULT = '../checkpoints/sessionwise'
### --- END default constants---

def train_step(loss, max_gradient_norm=1.0):
    global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 300, 0.96, staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

def train():
    # Set the random seeds for reproducibility.
    #tf.set_random_seed(42)
    #np.random.seed(42)
    
    snizer = Sessionizer()
    train_sess = snizer.get_sessions_with_numbers()
    snizer = Sessionizer('../data/val_session')
    val_sess = snizer.get_sessions_with_numbers()
    # Choose a random subset to validate on, because otherwise the validation takes too long
    val_sess = np.random.choice(val_sess, 50)
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, 300, FLAGS.num_layers)
    # Feeds for inputs.
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [FLAGS.padding,])
        dec_input = tf.placeholder(tf.int32, [FLAGS.padding,])
        target = tf.placeholder(tf.int32, [FLAGS.padding,])
        s0 = tf.placeholder(tf.float32, [1, FLAGS.s_dim])
    # Data pipeline
    logits, S = model.inference(query, dec_input, s0)
    loss = model.loss(logits, target)
    acc = model.accuracy(logits, target)
    # Initialize optimizer
    #opt = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    opt_operation = train_step(loss)
    print('[Model was created.]')
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
        if FLAGS.resume:# == 'True':
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            print('[Latest model was restored.]')
        else:
            sess.run(tf.global_variables_initializer()) 
            print('[Initialized variables.]')
#         Do a loop
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print('[%s: Starting training.]' % time)
        num_examples_seen = 0
        best_loss = 0
        for iteration in range(len(train_sess)):#range(FLAGS.max_steps):
            # Select a random session to train on.
            session = train_sess[iteration]#np.random.choice(train_sess)
            state = np.zeros((1,FLAGS.s_dim))
            #losses = []
            for i in range(len(session)-1):
                # Loop over the session and predict each query using the previous ones                
                x1 = pad_query(session[i], pad_size=FLAGS.padding)
                x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
                
                if i < len(session)-2:
                    state = sess.run(S, feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                else:
                    # We're at the anchor query of this session
                    _, l, summary = sess.run([opt_operation, loss, merged], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                    writer.add_summary(summary, iteration)
                #losses.append(l)
                num_examples_seen += 1
            # Append the loss of this session to the training data
            train_writer.append(l)#np.mean(losses))
            if (iteration+1) % FLAGS.print_freq == 0:
                print('Visited %s examples of %s sessions. Loss: %f' % (num_examples_seen, iteration+1, train_writer[-1]))
            # Evaluate the model
            if iteration % FLAGS.eval_freq == 0 or iteration == FLAGS.max_steps - 1:
                val_losses = []
                accs = []
                # Loop over all the sessions in the validation set
                for session in val_sess:
                    state = np.zeros((1,FLAGS.s_dim))
                    losses = []
                    for i in range(len(session)-1):
                        x1 = pad_query(session[i], pad_size=FLAGS.padding)
                        x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                        y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
                        
                        if i < len(session)-2:
                            state, l = sess.run([S, loss], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                        else:
                            state, l, accuracy = sess.run([S, loss, acc], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                            accs.append(accuracy)
                        losses.append(l)
                    val_losses.append(np.mean(losses))
                test_writer.append(np.mean(val_losses))
                print('-' * 40)
                print('Train loss at step %s: %f.' % (iteration, train_writer[-1]))
                print('Test loss at step %s: %f. Accuracy: %f' % (iteration, test_writer[-1], np.mean(accs)))
                print('Number of examples seen: %s' % num_examples_seen)
                print('-' * 40)
                # Save the loss data so that we can plot it later
                np.save(FLAGS.log_dir + '/train', np.array(train_writer))
                np.save(FLAGS.log_dir + '/test', np.array(test_writer))
                # Save the model
                if iteration == 0:
                    best_loss = test_writer[-1]
                if test_writer[-1] < best_loss:
                    best_loss = test_writer[-1]
                    file_name = FLAGS.checkpoint_dir + '/HRED_model'
                    saver.save(sess, file_name, iteration)
    writer.close()
    time = datetime.now().strftime('%d-%m %H:%M:%S')
    print('[%s: Training finished.]' % time)
    
def test():
    """
    This method calculates the mean loss and accuracy of the model
    """
    snizer = Sessionizer('../data/test_session')
    test_sess = snizer.get_sessions_with_numbers()
    # Create model
    model = HRED(FLAGS.vocab_dim, FLAGS.q_dim, FLAGS.s_dim, 300, FLAGS.num_layers)
    # Feeds for inputs.
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [FLAGS.padding,])
        dec_input = tf.placeholder(tf.int32, [FLAGS.padding,])
        target = tf.placeholder(tf.int32, [FLAGS.padding,])
        s0 = tf.placeholder(tf.float32, [1, FLAGS.s_dim])
    # Data pipeline
    logits, S = model.inference(query, dec_input, s0)
    loss = model.loss(logits, target)
    acc = model.accuracy(logits, target)
    # Create a saver.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:        
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
        print('Model was restored.')
        
        losses = []
        accuracies = []
        for j, session in enumerate(test_sess):
            # Encode the session
            state = np.zeros((1,FLAGS.s_dim))
            for i in range(len(session)-1):
                # Loop over the session and predict each query using the previous ones                
                x1 = pad_query(session[i], pad_size=FLAGS.padding)
                x2 = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='dec_input')
                y = pad_query(session[i+1], pad_size=FLAGS.padding, q_type='target')
                
                if i < len(session)-2:
                    state = sess.run(S, feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                else:
                    # We're at the anchor query of this session
                    l, accuracy = sess.run([loss, acc], feed_dict={query: x1, dec_input: x2, target: y, s0: state})
                    losses.append(l)
                    accuracies.append(accuracy)
            if (j+1)%FLAGS.print_freq == 0:
                print('-' * 40)
                print('After %s sessions:' % (j+1))
                print('Mean loss: %f' % np.mean(losses))
                print('Mean accuracy : %f' % np.mean(accuracies))            
    
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
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    if not FLAGS.is_train:# == 'False':
        print('Going to test latest model in directory %s' % FLAGS.checkpoint_dir)
        test()
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
    parser.add_argument('--is_train', type = str, default = False,
                      help='Training or feature extraction')
    parser.add_argument('--resume', type = str, default = False,
                      help='Resume training from latest checkpoint')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run()