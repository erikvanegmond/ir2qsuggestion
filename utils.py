import cPickle as pickle
import numpy as np

def create_word_mappings():
    print 'Loading aol_vocab.dict.pkl...'
    vocab = pickle.load(open('../data/aol_vocab.dict.pkl', 'rb'))
    print 'Creating mappings for %s words...' % len(vocab)
    word2index = {w:n1 for w, n1, n2 in vocab}
    index2word = {n1:w for w, n1, n2 in vocab}
    start_symbol_idx = max(index2word.keys())+1
    index2word[start_symbol_idx] = '<q>'
    word2index['<q>'] = start_symbol_idx
    print 'Saving to word2index.p...'
    pickle.dump(word2index, open('../data/word2index.p', 'wb'))
    print 'Saving to index2word.p...'
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
                aug_session.append(aug_query)
                queries += 1
            # Add session to data
            aug_data.append(aug_session)
            sessions_counter += 1
            # Store the data every 100.000 sessions.
            if sessions_counter % 100000 == 0:
                print '%s sessions. Parsed %d queries. %f sessions skipped.' % (sessions_counter, queries, bad_sessions)
                with open('../data/aug_'+ name + '.pkl', 'wb') as f:
                    pickle.dump(aug_data, f) 
        else:
            bad_sessions += 1
    return aug_data

def checkEqual(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)
