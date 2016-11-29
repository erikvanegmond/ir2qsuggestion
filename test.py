from datetime import datetime
from model import Model
from sessionizer import Sessionizer
import cPickle as pickle
import numpy as np

snizer = Sessionizer()
start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading sessions...]" % time)
with open('../data/aug_test_session.pkl', 'rb') as f:
    test_sessions = pickle.load(f)
print("[Loaded %s test sessions. It took %f seconds.]" % (len(test_sessions), (datetime.now() - start_time).seconds))

start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Loading model...]" % time)
m = Model.load('../models/29-11_4.589_0_90005x1000x90005.npz')
print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))

start_time = datetime.now()
time = start_time.strftime('%d-%m %H:%M:%S')
print("[%s: Calculating loss...]" % time)
losses = []
for i, session in enumerate(test_sessions):
    X_test = session[:-1]
    y_test = session[1:]
    loss = m.calculate_loss(X_test, y_test)
    losses.append(loss)
    if i % 10000 == 0:
        print('Loss after %s sessions is: %d' % (i, np.mean(losses)))
print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))

print("Mean loss: %s" % (np.mean(losses)))
print("Stddev loss: %s" % (np.std(losses)))