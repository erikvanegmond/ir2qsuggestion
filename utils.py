import pickle
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
    
def appent_start_stop_num(data, aug_data=None):
    word2index = pickle.load( open( "../data/word2index.p", "rb" ) )
    if aug_data == None:
        aug_data = np.array([])
    else:
        aug_data = aug_data
    queries = 0
    sessions = len(aug_data)
    
    for i in np.arange(len(aug_data), len(data)):
        session = data[i]
        aug_session = np.array([])
        for query in session:
            aug_query = np.append(np.append(word2index['<q>'], query), word2index['</q>'])
            aug_session = np.append(aug_session, aug_query)
            queries += 1
            if queries % 100000 == 0:
                print 'Added start- and stop symbols to %s queries and %d sessions.' % (queries, sessions)
                pickle.dump(aug_data, open('../data/augmented_data.p', 'wb'))
        aug_data = np.append(aug_data, aug_session)
        sessions += 1
    
    return aug_data