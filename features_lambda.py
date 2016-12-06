import cPickle
import numpy as np
import lambda_mart as lm
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

# Do LambdaMart for 3 different scenario's
# 1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:
def create_next_query_csv():
    print('[Creating dataset for next_query predictions.]')
    experiment_string = "next_query"
    print("Performing experiment: " + experiment_string)
    corresponding_queries = lm.next_query_prediction(lm.sessions, experiment_string)
    print("---" * 30)

## 2 RobustPrediction (when the context is perturbed with overly common queries)
## label 100 most frequent queries in the background set as noisy
def create_noisy_query_csv():
    experiment_string = "noisy"
    print("Performing experiment: " + experiment_string)
    noisy_query_sessions = lm.noisy_query_prediction()
    corresponding_queries_noisy = lm.next_query_prediction(noisy_query_sessions, experiment_string)
    print("---" * 30)

# 3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)
def create_longtail_query_csv():
    experiment_string = "long_tail"
    print("Performing experiment: " + experiment_string)
    corresponding_queries_lt = lm.make_long_tail_set(lm.sessions, experiment_string)
    print("---" * 30)

#def next_query_HRED_features():
#    hred = hredf.HRED()
#    adj = ad.ADJ()
#    pkl_file = open('../data/lm_tr_sessions.pkl', 'rb')
#    sessions = pkl.load(pkl_file)
#    df = pd.read_csv('../data/lamdamart_data_next_query.csv')
#    df['HRED'] = pd.Series(np.zeros(len(df)), index=df.index)
#    for i in range(len(sessions)):
#        session = sessions[i]
#        anchor_query = session[-2]
#        adj_dict = adj.adj_function(anchor_query)
#        highest_adj_queries = adj_dict['adj_queries']
#        hred_feats = hred.calculate_feature(anchor_query, highest_adj_queries)
#        df['HRED'][i:i+len(highest_adj_queries)] = hred_feats