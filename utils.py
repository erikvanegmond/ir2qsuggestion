import pickle
import numpy as np
import features.adj as adj
from datetime import datetime
#from model.model import Model

word2index = pickle.load( open( "../data/word2index.p", "rb" ) )
index2word = pickle.load( open( "../data/index2word.p", "rb" ) )
# Special vocabulary symbols - we always put them at the start.
_PAD = '<pad>'
_GO = '<go>'
_EOS = '<eos>'
_UNK = '<unk>'

PAD_ID = 1
GO_ID = 2
EOS_ID = 3
UNK_ID = 0

def create_word_mappings():
    print('Loading aol_vocab.dict.pkl...')
    vocab = pickle.load(open('../data/aol_vocab.dict.pkl', 'rb'))
    print('Creating mappings for %s words...' % len(vocab))
    word2index = {}
    index2word = {}
    for w, n1, n2 in vocab:
        if n1 == PAD_ID:
            word2index[_PAD] = PAD_ID
            index2word[PAD_ID] = _PAD
        elif n1 == GO_ID:
            word2index[_GO] = GO_ID
            index2word[GO_ID] = _GO
        elif n1 == EOS_ID:
            word2index[_EOS] = EOS_ID
            index2word[EOS_ID] = _EOS
        else:
            word2index[w] = n1
            index2word[n1] = w
    print('Saving to word2index.p...')
    pickle.dump(word2index, open('../data/word2index.p', 'wb'))
    print('Saving to index2word.p...')
    pickle.dump(index2word, open('../data/index2word.p', 'wb'))

def create_test_train(data, test_size):
    data = np.array(data)
    test_idx = np.random.choice(np.arange(len(data)), test_size)
    train_idx = np.setxor1d(np.arange(len(data)), test_idx)
    train = data[train_idx]
    test = data[test_idx]

    return test, train
    
def append_start_stop_num(sessions, name):
    word2index = pickle.load( open( "../data/word2index.p", "rb" ) )
    aug_data = []

    queries = 0
    bad_sessions = 0
    sessions_counter = 0
    # Loop over all sessions
    for i in np.arange(len(sessions)):
        session = sessions[i]
        # If all the queries in this session are the same, do not add.
        if not checkEqual(session):
            aug_session = []
            # Go over all queries
            for query in session:
                # Append the start and stop symbol (in indices)
                aug_query = np.append(np.append(word2index['<q>'], query), word2index['</q>'])
                # Add query to session
                aug_session.append(aug_query.astype(np.int32))
                queries += 1
            # Add session to data
            aug_data.append(aug_session)
            sessions_counter += 1
            # Store the data every 100.000 sessions.
            if sessions_counter % 100000 == 0:
                print('%s sessions. Parsed %d queries. %f sessions skipped.' % (sessions_counter, queries, bad_sessions))
        else:
            bad_sessions += 1
    with open('../data/aug_'+ name + '.pkl', 'wb') as f:
        pickle.dump(aug_data, f) 
    return aug_data

def checkEqual(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
    
def vectorify(string):
    vect = []
    for words in string.split():
        try:
            word = word2index[words]
        except KeyError:
            word = word2index['<unk>']
        vect.append(word)
    
    return np.array(vect, np.int32)
    
def create_feature_data():
    ADJ = adj.ADJ()
    
    start_time = datetime.now()
    time = start_time.strftime('%d-%m %H:%M:%S')
    print("[%s: Loading sessions lm_tr_sessions.pkl]" % time)
    pkl_file = open('../data/lm_tr_sessions.pkl', 'rb')
    sessions = pickle.load(pkl_file)
    pkl_file.close()
    print("[Loaded %s test sessions. It took %f seconds.]" % (len(sessions), (datetime.now() - start_time).seconds))
    
    start_time = datetime.now()
    time = start_time.strftime('%d-%m %H:%M:%S')
    print("[%s: Loading model...]" % time)
    m = Model.load('../models/29-11_4.589_0_90005x1000x90005.npz')
    print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
    
    features = {}
    queries = 0
    bad_sessions = 0
    
    start_time = datetime.now()
    time = start_time.strftime('%d-%m %H:%M:%S')
    print("[%s: Creating features...]" % time)
    for session in sessions:
        # Get the anchor queries
        anchor_query = session[-2]
        adj_dict = ADJ.adj_function(anchor_query)
        highest_adj_queries = adj_dict['adj_queries']
        features[anchor_query] = {}
        # Calculate the likelihood between the queries
        for sug_query in highest_adj_queries:
            num_anchor_query = vectorify(anchor_query)
            num_sug_query = vectorify(sug_query)                    
            likelihood = m.likelihood(num_anchor_query, num_sug_query)
            features[anchor_query][sug_query] = likelihood
        queries += 1
        if queries % 10000 == 0:
            print("[Visited %s anchor queries. %d sessions were skipped.]" % (queries, bad_sessions))
    print("[Saving features %s features.]" % (len(features)))
    pickle.dump(features, open('../data/HRED_features.pkl', 'wb'))
    print("[It took %d seconds.]" % ((datetime.now() - start_time).seconds))
