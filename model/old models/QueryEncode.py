import nltk
import csv
import itertools
import time
import numpy as np
import operator
import sys
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip


class QueryEnc:
    def __init__(self, vocab, hidden, bptt_truncate=-1):
        self.vocab = vocab
        self.hidden = hidden
        self.bptt_truncate = bptt_truncate
        invSqr1 = np.sqrt(1./vocab)
        invSqr2 = np.sqrt(1./hidden)
        E = np.random.uniform(-invSqr1, invSqr1, (hidden, vocabSize))
        U = np.random.uniform(-invSqr2, invSqr2, (6, hidden, hidden))
        V = np.random.uniform(-invSqr2, invSqr2, (vocab, hidden))
        W = np.random.uniform(-invSqr2, invSqr2, (6, hidden, hidden))
        b = np.zeros((6, hidden))
        c = np.zeros(vocab)
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.cachedE = theano.shared(name='cachedE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.cachedU = theano.shared(name='cachedU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.cachedV = theano.shared(name='cachedV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.cachedW = theano.shared(name='cachedW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.cachedb = theano.shared(name='cachedb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.cachedc= theano.shared(name='cachedc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E = self.E
        U = self.U
        V = self.V
        W = self.W
        b = self.b
        c = self.c
        x = T.ivector('x')
        y = T.ivector('y')

        def forward(x_data, prev_hidden_state1, prev_hidden_state2):
            x_embed = E[:, x_data]

            #Despicable Connection
            z = T.nnet.hard_sigmoid(U[0].dot(x_embed) + W[0].dot(prev_hidden_state1) + b[0])
            r = T.nnet.hard_sigmoid(U[1].dot(x_embed) + W[1].dot(prev_hidden_state1) + b[1])
            c_mem = T.tanh(U[2].dot(x_embed) + W[2].dot(prev_hidden_state1 * r) + b[2])
            hidden_state1 = (T.ones_like(z) - z) * c_mem + z * prev_hidden_state1

            #Despicable Connection: the sequel
            z = T.nnet.hard_sigmoid(U[3].dot(hidden_state1) + W[3].dot(prev_hidden_state2) + b[3])
            r = T.nnet.hard_sigmoid(U[4].dot(hidden_state1) + W[4].dot(prev_hidden_state2) + b[4])
            c_mem = T.tanh(U[5].dot(hidden_state1) + W[5].dot(prev_hidden_state2 * r) + b[5])
            hidden_state2 = (T.ones_like(z) - z) * c_mem + z * prev_hidden_state2
            output = T.nnet.softmax(V.dot(hidden_state2) + c)[0]

            return [output, hidden_state1, hidden_state2]

        [pred_output, s1, s2], updates = theano.scan(forward, sequences=x, truncate_gradient=self.bptt_truncate,
                                          outputs_info=[None, dict(initial=T.zeros(self.hidden)), 
                                                       dict(initial=T.zeros(self.hidden))])

        prediction = T.argmax(pred_output, axis=1)
        output_error = T.sum(T.nnet.categorical_crossentropy(pred_output, y))

        cost = output_error
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dV = T.grad(cost, V)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dc = T.grad(cost, c)

        self.predict = theano.function([x], pred_output)
        self.predict_query = theano.function([x], prediction)
        self.cross_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dV, dW, db, dc])

        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        temp = 1 - decay
        cachedE = decay * self.cachedE + (temp) * dE ** 2
        cachedU = decay * self.cachedU + (temp) * dU ** 2
        cachedV = decay * self.cachedV + (temp) * dV ** 2
        cachedW = decay * self.cachedW + (temp) * dW ** 2
        cachedb = decay * self.cachedb + (temp) * db ** 2
        cachedc = decay * self.cachedc + (temp) * dc ** 2

        self.sgd_step = theano.function([x, y, learning_rate, theano.In(decay,
                                                                        value=0.9)],
                                       [], updates = [(E, E - learning_rate * dE / T.sqrt(cachedE + 1e-6)),
                                                     (U, U - learning_rate * dU / T.sqrt(cachedU + 1e-6)),
                                                     (V, V - learning_rate * dV / T.sqrt(cachedV + 1e-6)),
                                                     (W, W - learning_rate * dW / T.sqrt(cachedW + 1e-6)),
                                                     (b, b - learning_rate * db / T.sqrt(cachedb + 1e-6)),
                                                     (c, c - learning_rate * dc / T.sqrt(cachedc + 1e-6)),
                                                     (self.cachedE, cachedE),
                                                     (self.cachedU, cachedU),
                                                     (self.cachedV, cachedV),
                                                     (self.cachedW, cachedW),
                                                     (self.cachedb, cachedb),
                                                     (self.cachedc, cachedc)])

    def total_loss(self, X, Y):
        return np.sum([self.cross_error(x, y) for x, y in zip(X, Y)])
    def calc_loss(self, X, Y):
        queries = np.sum([len(y) for y in Y])
        return self.total_loss(X, Y)/float(queries)

"""
Setting up global configurations. Vocab size is selected dynamically with
the number of unique queries. This should be clamped to made only n frequent
terms as single occurence will not impart any learnable features.
However, this is for testing the RNN.
"""

vocabSize = 6000
nonFrequent = "unk"
query_start = "Q"
query_end = "\Q"

"""
Simple preprocessing. Loading data and casting them to idx vec form.
For testing the RNN. Later need to put in 1-hot or even word2vec features
instead.
Loop over to load all queries from different files. Didn't use gzip.
Should be added.
"""

print "Loading query data"
a = time.time()
with open("AOL_search_data_leak_2006/Testset.txt") as file:
    reader = csv.reader(file, delimiter='\t')
    reader.next()
    queries_raw = itertools.chain(*[nltk.sent_tokenize(x[1].lower())for x in reader])
    queries = ["%s %s %s" % (query_start, x, query_end) for x in queries_raw]
print "Total queries %d in %f seconds" % (len(queries), (time.time() - a))

"""
breaking up things into tokens. Using NLTK, because it's easier. There are
pythonic ways, which we can implement later.
"""

queries_token = [nltk.word_tokenize(query) for query in queries]
counts = nltk.FreqDist(itertools.chain(*queries_token))
vocabSize = len(counts.items()) + 1
query_count = counts.most_common(vocabSize - 1)
idx_to_word = [x[0] for x in query_count]
idx_to_word.append(nonFrequent)
word_to_idx = dict([(x, i) for i, x in enumerate(idx_to_word)])

for i, query in enumerate(queries_token):
    queries_token[i] = [w if w in word_to_idx else nonFrequent for w in query]

print "\nExample sentence: %s" % queries[0]
print "Pre-processed: %s" % queries_token[0]
print list(query_count)[:5]
print "%d unique queries" % len(counts.items())

x_train = np.asarray([[word_to_idx[x] for x in query[:-1]] for query in queries_token])
y_train = np.asarray([[word_to_idx[x] for x in query[1:]] for query in queries_token])

print x_train[0]
print y_train[0]

model = QueryEnc(vocabSize, 100)

print "Beginning training"
for epoch in range(20):
    print "Epoch: %d" % (epoch)
    j = 1
    for i in np.random.permutation(len(y_train)):
        #print j, "...",
        model.sgd_step(x_train[i], y_train[i], 0.001, 0.9)
        j = j + 1
        if (j % 60 == 0):
            loss = model.calc_loss(x_train[:1000], y_train[:1000])
            print "%d th training. Loss: %f" % (j, loss)
    print "======================================================="
    sys.stdout.flush()

