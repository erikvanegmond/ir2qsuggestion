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
    for query in suggestions[1:]:
        prob, prev_state = m.likelhood(query, prev_state)
        prob_list.append(prob)
    return prob_list

def aug_data(data):
    word2index = cPickle.load(open('../data/word2index.pkl', 'rb'))
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
def create_next_query_csv(sessions):
    print('[Creating dataset for next_query predictions.]')
    experiment_string = "next_query_val"
    print("Performing experiment: " + experiment_string)
    lm.next_query_prediction(sessions, experiment_string)
    print("---" * 30)
    lm.hred.save()

## 2 RobustPrediction (when the context is perturbed with overly common queries)
## label 100 most frequent queries in the background set as noisy
def create_noisy_query_csv(sessions):
    experiment_string = "noisy_val"
    print("Performing experiment: " + experiment_string)
    noisy_query_sessions = lm.noisy_query_prediction(sessions)
    lm.next_query_prediction(noisy_query_sessions, experiment_string)
    print("---" * 30)
    lm.hred.save()

# 3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)
def create_longtail_query_csv(sessions1, experiment_string):
    # experiment_string = "long_tail_val"
    print("Performing experiment: " + experiment_string)
    lm.make_long_tail_set(sessions1, experiment_string)
    print("---" * 30)
    lm.hred.save()

print('[Creating next_query features.]')
# create_next_query_csv()
# create_noisy_query_csv()
sessions_tr = lm.adj.find_suitable_sessions("../data/lm_train_sessions.pkl")
create_next_query_csv(sessions=sessions_tr)
sessions_val = lm.adj.find_suitable_sessions("../data/lm_val_sessions.pkl")
create_next_query_csv(sessions=sessions_val)
sessions_test = lm.adj.find_suitable_sessions("../data/lm_test_sessions.pkl")
create_next_query_csv(sessions=sessions_test)

print('[Creating noisy_query features.]')
# create_next_query_csv()
# create_noisy_query_csv()
#sessions_tr = lm.adj.find_suitable_sessions("../data/lm_train_sessions.pkl")
create_noisy_query_csv(sessions=sessions_tr)
#sessions_val = lm.adj.find_suitable_sessions("../data/lm_val_sessions.pkl")
create_noisy_query_csv(sessions=sessions_val)
#sessions_test = lm.adj.find_suitable_sessions("../data/lm_test_sessions.pkl")
create_noisy_query_csv(sessions=sessions_test)
    
print('[Creating long_tail features.]')
# create_next_query_csv()
# create_noisy_query_csv()
#sessions_tr = lm.adj.find_suitable_sessions("../data/lm_train_sessions.pkl")
create_longtail_query_csv(sessions=sessions_tr,experiment_string="long_tail_train")
#sessions_val = lm.adj.find_suitable_sessions("../data/lm_val_sessions.pkl")
create_longtail_query_csv(sessions=sessions_val,experiment_string="long_tail_val")
#sessions_test = lm.adj.find_suitable_sessions("../data/lm_test_sessions.pkl")
create_longtail_query_csv(sessions=sessions_test,experiment_string="long_tail_test")