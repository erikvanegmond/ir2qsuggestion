import numpy as np

import logging

from sklearn.model_selection import cross_val_score

from rankpy.queries import Queries
from rankpy.queries import find_constant_features

from rankpy.models import LambdaMART

import bisect
import random
from collections import Counter, Sequence


def lambdaMart(data):
    # Turn on logging.
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    # divide set into train, val and test set
    # 55% train
    # 20% validation
    # 25% test
    training_queries, validation_queries, test_queries = np.split(data.sample(frac=1), [int(.55 * len(data)), int(.75 * len(data))])

    logging.info('================================================================================')

    # Save them to binary format ...
    training_queries.save('data/MQ2007/Fold1/training')
    validation_queries.save('data/MQ2007/Fold1/validation')
    test_queries.save('data/MQ2007/Fold1/test')

    # ... because loading them will be faster.
    training_queries = Queries.load('data/MQ2007/Fold1/training')
    validation_queries = Queries.load('data/MQ2007/Fold1/validation')
    test_queries = Queries.load('data/MQ2007/Fold1/test')

    logging.info('================================================================================')

    # Print basic info about query datasets.
    logging.info('Train queries: %s' % training_queries)
    logging.info('Valid queries: %s' % validation_queries)
    logging.info('Test queries: %s' %test_queries)

    logging.info('================================================================================')

    # Print basic info about query datasets.
    logging.info('Train queries: %s' % training_queries)
    logging.info('Valid queries: %s' % validation_queries)
    logging.info('Test queries: %s' %test_queries)

    logging.info('================================================================================')

    # Set this to True in order to remove queries containing all documents
    # of the same relevance score -- these are useless for LambdaMART.
    remove_useless_queries = False

    # Find constant query-document features.
    cfs = find_constant_features([training_queries, validation_queries, test_queries])

    # Get rid of constant features and (possibly) remove useless queries.
    training_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
    validation_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
    test_queries.adjust(remove_features=cfs)

    # Print basic info about query datasets.
    logging.info('Train queries: %s' % training_queries)
    logging.info('Valid queries: %s' % validation_queries)
    logging.info('Test queries: %s' % test_queries)

    logging.info('================================================================================')

    model = LambdaMART(metric='nDCG@10', max_leaf_nodes=7, shrinkage=0.1,
                       estopping=50, n_jobs=-1, min_samples_leaf=50,
                       random_state=42)

    model.fit(training_queries, validation_queries=validation_queries)

    logging.info('================================================================================')

    logging.info('%s on the test queries: %.8f'
                 % (model.metric, model.evaluate(test_queries, n_jobs=-1)))

    model.save('LambdaMART_L7_S0.1_E50_' + model.metric)

def ADJ_function(anchor_query):
    ADJ_results_list = cooccurence_based_ranking(anchor_query)
    best_20 = sorted(ADJ_results_list)
    return best_20
# Do LambdaMart for 3 different scenario's
#1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:

# instances,~ = data.shape
# one_hot_session_vector = np.zeros(instances)

def next_query_prediction(sessions):
    used_sessions = []
    for i,session in enumerate(sessions):
        session_length = len(session)
        # get anchor query and target query from session
        anchor_query = session[session_length-1]
        target_query = session[session_length]
        # extract 20 queries with the highest ADJ score (most likely to follow the anchor query in the data)
        highest_ADJ_queries = ADJ_function(anchor_query)

        # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
        if target_query in highest_ADJ_queries:
            # one_hot_session_vector[i] = 1

            # then add the session to the train, val, test data
            used_sessions.append(session)

data_next_query = next_query_prediction(sessions)
lambdaMart(data_next_query)

#2 RobustPrediction (when the context is perturbed with overly common queries)
# label 100 most frequent queries in the background set as noisy
def count_query_frequency(query_list):
    count_list = count(query_list)
    return count_list

def take_100_most_frequent(query_frequency_list):
    highest_100 = np.sort(query_frequency_list)[:100]
    return highest_100

def calculate_noise_prob(highest_100):
    noise_prob = highest_100[2]/sum(highest_100[2])
    highest_100[2] = noise_prob
    return highest_100

def get_random_noise(highest_100_prob):
    total = sum(w for c, w in highest_100_prob)
    r = random.uniform(0, total)
    upto = 0
    for c, w in highest_100_prob:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

#make list of all queries
query_list = [session[:] for session in sessions]
query_frequency_list = count_query_frequency(query_list)
highest_100 = take_100_most_frequent(query_frequency_list)
highest_100_prob = calculate_noise_prob(highest_100)
for session in sessions:
    # for each entry in the training, val and test set insert noisy query at random position
    random_place = np.random.randint(0,len(session))
    # probability of sampling a noisy query is proportional to frequency of query in background set
    noise = get_random_noise(highest_100_prob)
    session[random_place] = noise

data_noisy = next_query_prediction(sessions)
lambdaMart(data_noisy)


#3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)

# How?? Based on term occurence? Random? Last one?
def shorten_query(query):
    term = choose_term_to_get_rid_of(query)
    query.remove(term)
    return query

def make_long_tail_set(sessions):
    used_sessions = []
    for session in sessions:
        session_length = len(session)
        # get anchor query and target query from session
        anchor_query = session[session_length - 1]
        target_query = session[session_length]
        # Cannot use ADJ
        # Therefore iteratively shorten anchor query by dropping terms until we have a query that appears in the Background data
        for i in range(len(anchor_query)):
            if anchor_query not in background_set:
                anchor_query = shorten_query(anchor_query)
            else:
                # If match found, proceed as described in next-query prediction
                highest_ADJ_queries = ADJ_function(anchor_query)
                # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
                if target_query in highest_ADJ_queries:
                # one_hot_session_vector[i] = 1
                # then add the session to the train, val, test data
                used_sessions.append(session)
    return used_sessions

data_long_tail = make_long_tail_set(sessions)
lambdaMart(data_long_tail)
