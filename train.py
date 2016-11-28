import numpy as np
import sys
import utils
from datetime import datetime
from model import Model
from sessionizer import Sessionizer
import cPickle as pickle

print(chr(27) + "[2J")
snizer = Sessionizer()
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading sessions...]" % time)
#with open('../data/augmented_data.pkl', 'rb') as f:
#    sessions = pickle.load(f)
sessions = snizer.get_sessions_with_numbers()
#sessions = utils.append_start_stop_num(sessions)
#print("[Loaded %s sessions. It took %d seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
#test_size = 2
#test, train = utils.create_test_train(sessions, test_size)
#print("[Created the test (%s) and train (%d) sets...]" % (len(test), len(train)))
#sys.stdout.flush()

#from data import train_x, train_y, test_x, test_y, index2word
#index2word = pickle.load( open( "../data/index2word.p", "rb" ) )

HIDDEN_SIZE = 1000

#print(len(train_x), len(test_x), len(train_y), len(test_y))

def train_with_sgd(m, learning_rate=0.01, evaluate_loss_after=10):
    # We keep track of the losses so we can plot them later
    losses = [( 0, 1000.0 )]
    num_examples_seen = 0
    sessions_seen = 0

    print("Beginning training...")
    while losses[-1][1] > 0.1:
        
        # Optionally evaluate the loss
        if (sessions_seen % evaluate_loss_after == 0):
            print("[Calculating test loss...]")
            loss = 0
            for session in test:
                X_test = session[:-1]
                y_test = session[1:]
                loss += m.calculate_loss(X_test, y_test)
            loss /= len(test)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print("")
            print("[%s: Loss after %d examples = %f]" % (time, num_examples_seen, loss))
            sys.stdout.flush()
            if (len(losses) > 1 and losses[-2][1] - losses[-1][1] >= 0.1):
                save_path = "../models/" + datetime.now().strftime('%d-%m_') + str(loss)[:5]
                Model.save(m, save_path)

            for x in np.random.choice(test):

                model_out = m.predict_class(x)
                out = [index2word[i] for i in model_out]
                inp = [index2word[i] for i in x]
                trans = "%s =>\n   %s" % (" ".join(inp), " ".join(out))

                print(" * " + trans)

            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
        
        start_time = datetime.now()
        time = start_time.strftime('%d-%m %H:%M:%S')
        print("[%s: Starting epoch %d...]" % (time, sessions_seen))
        
        session = np.random.choice(train)        
        X = session[:-1]
        y = session[1:]
        # For each training example...
        ecount = 0
        for x, y in zip(X, y):
            m.SGD(x, y, learning_rate)
            num_examples_seen += 1
            ecount += 1
            if ecount % 1000 == 0:
                print(".",)
                sys.stdout.flush()
        sessions_seen += 1
        print("[Visited %s examples. It took %d seconds.]" % (ecount, (datetime.now() - start_time).seconds))
        sys.stdout.flush()

#start_time = datetime.now()
#time = start_time.strftime('%d-%m %H:%M:%S')
#print("[%s: Creating model...]" % time)
#m = Model(len(index2word), HIDDEN_SIZE, len(index2word))
#print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
#
##start_time = datetime.now()
##time = start_time.strftime('%d-%m %H:%M:%S')
##print("[%s: Loading model...]" % time)
##m = Model.load('../models/26-11_11.40_90004x1000x90004.npz')
##print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
#
#train_with_sgd(m)

