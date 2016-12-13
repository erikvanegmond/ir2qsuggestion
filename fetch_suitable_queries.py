from features.adj import ADJ
from features.ranker import Ranker
from datetime import datetime

start = datetime.now()
time = start.strftime('%d-%m %H:%M:%S')
print('[%s: Creating ADJ...]' % time)
# the train_sessions_file is the dataset that is used to find the suitable queries for. Sadly we chose an unfortunate name.
ranker = Ranker(train_sessions_file='../data/test_session')
adj = ADJ()
print('[It took %s seconds.]' % (start - datetime.now()).seconds)

start = datetime.now()
time = start.strftime('%d-%m %H:%M:%S')
print('[%s: Searching for suitable train queries...]' % time)
adj.find_suitable_sessions()
print('[It took %s seconds.]' % (start - datetime.now()).seconds)
