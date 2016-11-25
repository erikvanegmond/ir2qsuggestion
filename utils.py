import pickle
import numpy as np

def create_word_mappings():
    print 'Loading aol_vocab.dict.pkl...'
    vocab = pickle.load(open('../data/aol_vocab.dict.pkl', 'rb'))
    print 'Creating mappings for %s words...' % len(vocab)
    word2index = {w:n1 for w, n1, n2 in vocab}
    index2word = {n1:w for w, n1, n2 in vocab}
    print 'Saving to word2index.p...'
    pickle.dump(word2index, open('../data/word2index.p', 'wb'))
    print 'Saving to index2word.p...'
    pickle.dump(index2word, open('../data/index2word.p', 'wb'))