import cPickle
import numpy as np
import lambda_mart as lm

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

sessions = lm.adj.find_suitable_sessions("../data/lm_train_sessions.pkl")
# Do LambdaMart for 3 different scenario's
# 1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:
def create_next_query_csv():
    print('[Creating dataset for next_query predictions.]')
    experiment_string = "next_query_test"
    print("Performing experiment: " + experiment_string)
    lm.next_query_prediction(sessions, experiment_string)
    print("---" * 30)
    lm.hred.save(data_file="../data/HRED_features_test_next_query.pkl")

## 2 RobustPrediction (when the context is perturbed with overly common queries)
## label 100 most frequent queries in the background set as noisy
def create_noisy_query_csv():
    experiment_string = "noisy_test"
    print("Performing experiment: " + experiment_string)
    noisy_query_sessions = lm.noisy_query_prediction(sessions)
    lm.next_query_prediction(noisy_query_sessions, experiment_string)
    print("---" * 30)
    lm.hred.save(data_file="../data/HRED_features_test_noisy.pkl")

# 3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)
def create_longtail_query_csv():
    experiment_string = "long_tail_test"
    print("Performing experiment: " + experiment_string)
    lm.make_long_tail_set(sessions, experiment_string)
    print("---" * 30)
    lm.hred.save(data_file="../data/HRED_features_test_longtail.pkl")

print('[Creating test features.]')

create_longtail_query_csv()