import numpy as np

def find_Likelihood(m, anchor, tail):
    anchor = vectorify(anchor)
    tail = vectorify(tail)
    return m.likelhood(anchor, tail) 

def vectorify(string):
    vect = []
    for words in anchor:
        vect.append(word2index[words])
    vect = np.append(np.append(word2index['<q>'], vect), word2index['</q>'])
    return vect
