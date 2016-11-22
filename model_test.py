"""
@author: Jaimy
"""
import numpy as np
import theano
import theano.tensor as T

from GRU_RNN import Query_GRU, Session_GRU
from decoder import Decoder

class Model():
    
    def __init__(self, vocab, max_length, q_dim=1000, s_dim=1500, o_dim=300, bptt_truncate=4):
        self.params = []
        self.cache = []
        self.q_dim = q_dim
        self.o_dim = o_dim
        self.s_dim = s_dim
        self.vocab = vocab
        self.max_length = max_length
        self.bptt_truncate = bptt_truncate
        
        # Query encoding parameters
        invSqr1 = np.sqrt(1./len(vocab))
        invSqr2 = np.sqrt(1./o_dim)
        invSqr3 = np.sqrt(1./q_dim)
        Eq = np.random.uniform(-invSqr1, invSqr1, (q_dim, len(vocab)))
        Uq = np.random.uniform(-invSqr3, invSqr3, (3, q_dim, q_dim))
        Wq = np.random.uniform(-invSqr3, invSqr3, (3, q_dim, q_dim))
        bq = np.zeros((3, q_dim))
        # Theano: Created shared variables
        self.Eq = self.add_to_params(theano.shared(name='query/E', value=Eq.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/query/E', value=np.zeros(Eq.shape).astype(theano.config.floatX)))
        self.Uq = self.add_to_params(theano.shared(name='query/U', value=Uq.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/query/U', value=np.zeros(Uq.shape).astype(theano.config.floatX)))
        self.Wq = self.add_to_params(theano.shared(name='query/W', value=Wq.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/query/W', value=np.zeros(Wq.shape).astype(theano.config.floatX)))
        self.bq = self.add_to_params(theano.shared(name='query/b', value=bq.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/query/b', value=np.zeros(bq.shape).astype(theano.config.floatX)))
        
        # Session encoding parameters
        invSqr1 = np.sqrt(1./q_dim)
        invSqr2 = np.sqrt(1./s_dim)
        Us = np.random.uniform(-invSqr1, invSqr1, (3, s_dim, q_dim))
        Ws = np.random.uniform(-invSqr2, invSqr2, (3, s_dim, s_dim))
        bs = np.zeros((3, s_dim))
        # Theano: Created shared variables
        self.Us = self.add_to_params(theano.shared(name='session/U', value=Us.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/session/U', value=np.zeros(Us.shape).astype(theano.config.floatX)))
        self.Ws = self.add_to_params(theano.shared(name='session/W', value=Ws.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/session/W', value=np.zeros(Ws.shape).astype(theano.config.floatX)))
        self.bs = self.add_to_params(theano.shared(name='session/b', value=bs.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/session/b', value=np.zeros(bs.shape).astype(theano.config.floatX)))
        
        # Decoder parameters
        invSqr1 = np.sqrt(1./len(vocab))
        invSqr2 = np.sqrt(1./q_dim)
        D0 = np.random.uniform(-invSqr2, invSqr2, (q_dim, s_dim))
        b0 = np.zeros((q_dim, 1))
        invSqr3 = np.sqrt(1./o_dim)
        Ho = np.random.uniform(-invSqr3, invSqr3, (o_dim, q_dim))
        Eo = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        bo = np.zeros((o_dim, 1))
        O = np.random.uniform(-invSqr3, invSqr3, (o_dim, len(vocab)))
        Ud = np.random.uniform(-invSqr1, invSqr1, (3, q_dim, len(vocab)))
        Wd = np.random.uniform(-invSqr2, invSqr2, (3, q_dim, q_dim))
        bd = np.zeros((q_dim, 3))
        # Theano: Created shared variables
        self.D0 = self.add_to_params(theano.shared(name='decoder/D0', value=D0.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/D0', value=np.zeros(D0.shape).astype(theano.config.floatX)))
        self.b0 = self.add_to_params(theano.shared(name='decoder/b0', value=b0.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/b0', value=np.zeros(b0.shape).astype(theano.config.floatX)))
        self.Ho = self.add_to_params(theano.shared(name='decoder/Ho', value=Ho.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/Ho', value=np.zeros(Ho.shape).astype(theano.config.floatX)))
        self.Eo = self.add_to_params(theano.shared(name='decoder/Eo', value=Eo.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/Eo', value=np.zeros(Eo.shape).astype(theano.config.floatX)))
        self.bo = self.add_to_params(theano.shared(name='decoder/bo', value=bo.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/bo', value=np.zeros(bo.shape).astype(theano.config.floatX)))
        self.O = self.add_to_params(theano.shared(name='decoder/O', value=O.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/O', value=np.zeros(O.shape).astype(theano.config.floatX)))
        self.Ud = self.add_to_params(theano.shared(name='decoder/U', value=Ud.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/U', value=np.zeros(Ud.shape).astype(theano.config.floatX)))
        self.Wd = self.add_to_params(theano.shared(name='decoder/W', value=Wd.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/W', value=np.zeros(Wd.shape).astype(theano.config.floatX)))
        self.bd = self.add_to_params(theano.shared(name='decoder/b', value=bd.astype(theano.config.floatX)))
        self.cache.append(theano.shared(name='cached/decoder/b', value=np.zeros(bd.shape).astype(theano.config.floatX)))
        
        self.theano = {}
        self.__theano_build__()
        
    def __theano_build__(self):        
        
        def query_step(x_t, q_t_prev):
            E, U, W, b = self.Eq, self.Uq, self.Wq, self.bq
            x_emb = E[:,x_t]
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_emb) + W[0].dot(q_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_emb) + W[1].dot(q_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_emb) + W[2].dot(q_t_prev * r_t) + b[2])
            q_t = (T.ones_like(z_t) - z_t) * c_t + z_t * q_t_prev
    
            return [q_t]
        
        def session_step(q_t, s_t_prev):
            U, W, b = self.Us, self.Ws, self.bs
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(q_t) + W[0].dot(s_t_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(q_t) + W[1].dot(s_t_prev) + b[1])
            c_t = T.tanh(U[2].dot(q_t) + W[2].dot(s_t_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev
    
            return [s_t]
        
        def decoder_step(w_t_prev, d_t_prev):
            U, W, b = self.Ud, self.Wd, self.bd
            Ho, Eo, bo, O = self.Ho, self.Eo, self.bo, self.O
            # Create the activation of the current word
            omega = Ho.dot(d_t_prev) + Eo.dot(w_t_prev) + bo
            w_t = T.nnet.softmax((O.T).dot(omega)) # Should return a vector of the length of the vocabulary with probs for each word
            
            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(w_t) + W[0].dot(d_t_prev) + b[:,0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(w_t) + W[1].dot(d_t_prev) + b[:,1])
            c_t = T.tanh(U[2].dot(w_t) + W[2].dot(d_t_prev * r_t) + b[:,2])
            d_t = (T.ones_like(z_t) - z_t) * c_t + z_t * d_t_prev

            return [w_t, d_t]
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        q_0 = np.zeros(self.q_dim).astype(theano.config.floatX)
        Q, updates = theano.scan(query_step, sequences=x, truncate_gradient=self.bptt_truncate, outputs_info=[q_0])
        
        s_0 = np.zeros(self.s_dim).astype(theano.config.floatX)
        S, updates = theano.scan(session_step, sequences=Q, truncate_gradient=self.bptt_truncate, outputs_info=[s_0])
        
        h_0 = T.tanh(self.D0.dot(S[-1]) + self.b0) # Initialize the first recurrent activation with the session
        w_0 = np.zeros((len(self.vocab), 1)).astype(theano.config.floatX) # The length of the vocab, with 0 probability for every word        
        [W, D], updates = theano.scan(decoder_step, n_steps=self.max_length, truncate_gradient=self.bptt_truncate,
                                    outputs_info=[w_0, h_0])
        
        prediction = T.argmax(W, axis=1)
        loss = T.sum(T.nnet.categorical_crossentropy(prediction, y))
        
        dparams = T.grad(loss, self.params)
        
        self.predict = theano.function([x], W)
        self.predict_query = theano.function([x], prediction)
        self.cross_error = theano.function([x, y], loss)
        self.bptt = theano.function([x, y], dparams)
        
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        temp = 1 - decay
        cache = decay * self.cache + temp * np.power(dparams, 2)
        
        self.sgd_step = theano.function([x, y, learning_rate, theano.In(decay, value=0.9)], [], 
                                         updates = [(self.params, np.subtract(self.params, learning_rate * dparams / T.sqrt(np.add(cache, 1e-6)))),
                                                    (self.cache, cache)])

    def add_to_params(self, new_param):
        self.params.append(new_param)
        return new_param
#%%
vocabSize = 6000
numSessions = 100
queriesPerSession = 5
queryLength = 4
max_length = 10

def generate_query(vocab):
    queryLen = min(max(int(np.random.normal(queryLength, 2)), 1), max_length)
    query = np.append(np.random.choice(vocab, queryLen), np.zeros(max_length - queryLen).astype(int))
    #query = np.zeros((len(vocab), max_length))
    #query[idx, np.arange(max_length)] = 1
    return query

vocab = np.arange(vocabSize)
sessions = []
for s in np.arange(numSessions):
    numQueries = int(np.random.normal(queriesPerSession, 2))
    queries = [generate_query(vocab) for q in np.arange(numQueries)]
    sessions.append(queries)
    
train_perc = int(0.8 * len(sessions))
X_data = sessions[:train_perc]
X_test = sessions[train_perc:]

model = Model(vocab, 4, max_length)
#%%
iterations = 100
val_iter = 10

for iteration in iterations:
    X = np.random.choice(X_data)
    y = X[1:]
    X = X[:-1]
    model.sgd_step(X, y, 0.001, 0.9)
    if iteration % val_iter == 0:
        loss = model.calc_loss(X, y)
        print "%d th training. Loss: %f" % (iteration, loss)
        print "======================================================="