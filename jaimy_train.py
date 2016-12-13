from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
import pickle as pkl

from RNNTensors.jaimy_model import HRED
from sessionizer import Sessionizer

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs'
CHECKPOINT_DIR_DEFAULT = './checkpoints'
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
    #snizer = Sessionizer('../data/val_session')
    #val_sess = snizer.get_sessions_with_numbers()
    vocab = pkl.load(open('../data/aol_vocab.dict.pkl', 'rb'))
    # Create model
    model = HRED(len(vocab), 1000, 1500, 1)
    print('[Model was created.]')
    # Feeds for inputs.
#    inputs = []
#    targets = []
    with tf.variable_scope('input'):
        query = tf.placeholder(tf.int32, [10, 2, 1])
        target = tf.placeholder(tf.int32, [10, 1, 1])
#        for i in range(_buckets[-1][0]):  # Last bucket is the biggest one.
#            inputs.append(tf.placeholder(tf.int32, shape=[None], name="input{0}".format(i)))
#        for i in range(_buckets[-1][1] + 1):
#            targets.append(tf.placeholder(tf.int32, shape=[None], name="target{0}".format(i)))
    # Data pipeline
#    logits = model.inference([query], [target])
    logits = model.inference(tf.unstack(query, axis=0), [target])
    loss = model.loss(logits, [target])
    # Initialize optimizer
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    opt_operation = opt.minimize(loss)
    #opt_operation = train_step(losses)
    # Create a saver.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Initialize summary writers
        merged = tf.merge_all_summaries()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        # Initialize variables
#        sess.run(tf.global_variables_initializer())     
        # Do a loop
        print('[Starting training.]')
        dummy_set = train_sess[0]
        dummy_sess = [dummy_set] * 10
        for iteration in range(FLAGS.max_steps):
            
#            session = train_sess[0] #np.random.choice(train_sess)
            sess.run(tf.global_variables_initializer())
            x = []
            y = []
            for pairs in dummy_sess:
                x.append(np.array(pairs[-2]).reshape((2,1)))
                y.append(np.array(pairs[-1]).reshape((1,1)))
            
#            print(len(session[-1]))
#            x, y = np.array(session[-2]).reshape((2,1)), np.array(session[-1]).reshape((1,1))
            

#                # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
#                encoder_size, decoder_size = self.buckets[bucket_id]
#            input_feed = {}
#            for l in xrange(encoder_size):
#              input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
#            for l in xrange(decoder_size):
#              input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
#              input_feed[self.target_weights[l].name] = target_weights[l]
            # sess.run(y, feed_dict={i: d for i, d in zip(inputs, data)})
            
            _ = sess.run([opt_operation], feed_dict={query: x, target: y})
            #train_writer.add_summary(summ, iteration)
#        else:
#        _ = sess.run([opt_operation], feed_dict={x: x_data, y: y_data})
#        
            if iteration % FLAGS.eval_freq == 0 or iteration == FLAGS.max_steps - 1:
                print('%s iterations' % iteration)
#                x_val, y_val = cifar10.test.images, cifar10.test.labels
#        
#                summary, accuracy_test = sess.run([merged, acc], feed_dict={x: x_val, y: y_val})
#                test_writer.add_summary(summary, iteration)
#                
#                print('Test accuracy at step %s: %s' % (iteration, accuracy_test))
#        
#            if iteration % FLAGS.checkpoint_freq == 0 or iteration == FLAGS.max_steps - 1:
#                file_name = FLAGS.checkpoint_dir + '/model.ckpt'
#                saver.save(sess, file_name)
    train_writer.close()
    test_writer.close()

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
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
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