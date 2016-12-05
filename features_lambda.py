import cPickle
import numpy as np
import pandas as pd
import features.HRED as hredf
import cPickle as pkl
import features.adj as ad

def getFeatures(m, anchor, suggestions):
    initial = m.init_rep
    suggestions = aug_data(suggestions)
    prob, primed_state = m.likelhood(anchor, initial)
    prob_list = []
    prob, prev_state = m.likelhood(suggestions[0], primed_state)
    prob_list.append(prob)
    for query in suggestion[1:]:
        prob, prev_state = m.likelhood(query, prev_state)
        prob_list.append(prob)
    return prob_list

def aug_data(data):
    word2index = cPickle.load(open('../data/word2index.p', 'rb'))
    auged_data = []
    for suggestion in data:
        query = word2index[suggestion]
        tmp = np.append(np.append(word2index['<q>'], query),
                        word2index['</q>'])
        auged_data.append(tmp.astype(np.int32))
    return auged_data

def next_query_HRED_features():
    hred = hredf.HRED()
    adj = ad.ADJ()
    pkl_file = open('../data/lm_tr_sessions.pkl', 'rb')
    sessions = pkl.load(pkl_file)
    df = pd.read_csv('../data/lamdamart_data_next_query.csv')
    df['HRED'] = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(len(sessions)):
        session = sessions[i]
        anchor_query = session[-2]
        adj_dict = adj.adj_function(anchor_query)
        highest_adj_queries = adj_dict['adj_queries']
        hred_feats = hred.calculate_feature(anchor_query, highest_adj_queries)
        df['HRED'][i:i+len(highest_adj_queries)] = hred_feats