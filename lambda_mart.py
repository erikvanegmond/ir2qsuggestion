#from main

import pandas as pd
import os
data_path = "../AOL-user-ct-collection/"

# my own
import numpy as np

import logging

from sklearn.model_selection import cross_val_score

from rankpy.queries import Queries
from rankpy.queries import find_constant_features

from rankpy.models import LambdaMART

from collections import Counter

import operator

# import main
import random

def read_files(max_files=10):  # -> pd.DataFrame:
    print "reading {} files".format(max_files)
    file_counter = 0
    df = pd.DataFrame()
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            df = pd.concat(
                [df, pd.read_csv(file_path, sep="\t", parse_dates=[2], infer_datetime_format=True, usecols=[0, 1, 2])])
            file_counter += 1
    return df


def lambdaMart(data):
    # Turn on logging.
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    # divide set into train, val and test set
    # 55% train
    # 20% validation
    # 25% test
    training_queries, validation_queries, test_queries = np.split(data.sample(frac=1), [int(.55 * len(data)), int(.75 * len(data))])

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

# Made by Erik
#we have options that could be the target query
#we need to have the 20 most cooccurent
def ADJ_function(anchor_query, sessions):
    cooccurence_list = []
    for sess in sessions:
        sess = sess.values
        anchor_occurence = np.where(sess == anchor_query)[0]
        if anchor_occurence.size:
            for index in anchor_occurence:
                if index+1 < len(sess):
                    print(sess[index+1])
                    cooccurence_list.append(sess[index+1])
                else:
                    continue
    print("cooccurence: " +  str(cooccurence_list))
    freq_dict = Counter(cooccurence_list)
    sorted_ADJ_results = sorted(freq_dict.items(), key=operator.itemgetter(1))
    # should be 20, when not in test fase
    best_20 = map(operator.itemgetter(0), sorted_ADJ_results)[:len(sorted_ADJ_results)]
    sugg_features = map(operator.itemgetter(1), sorted_ADJ_results)[:len(sorted_ADJ_results)]
    print("sugg_features" + str(sugg_features))
    return best_20, sugg_features

# Do LambdaMart for 3 different scenario's
# 1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:

# instances,~ = data.shape
# one_hot_session_vector = np.zeros(instances)

def next_query_prediction(anchor_query, sessions):
    lambdamart_data = []
    corresponding_queries = []
    for i,session in enumerate(sessions):
        # get anchor query and target query from session
        anchor_query = session.iloc[-2]
        # Done for testing, will be the result from the RNN
        target_query = session.iloc[-1]
        # extract 20 queries with the highest ADJ score (most likely to follow the anchor query in the data)
        highest_ADJ_queries, sugg_features = ADJ_function(anchor_query, sessions)

        # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
        if target_query in highest_ADJ_queries:
            target_vector = -1 * np.ones(len(highest_ADJ_queries))
            target_query_index = np.where(highest_ADJ_queries == target_query)[0]
            target_vector[target_query_index] = 1
            # print(target_vector)
            # then add the session to the train, val, test data
            for j,query in enumerate(highest_ADJ_queries):
                lambdamart_data.append([target_vector[j], sugg_features[j]])
                corresponding_queries.append(highest_ADJ_queries[j])
        lambdamart_df = pd.DataFrame(lambdamart_data)
        print(lambdamart_df)
        results = lambdaMart(lambdamart_df)
        print(results)
    return results, corresponding_queries



# 2 RobustPrediction (when the context is perturbed with overly common queries)
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

# 3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)

# How?? Based on term occurrence? Random? Last one?
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
        # Therefore iteratively shorten anchor query by dropping terms
        # until we have a query that appears in the Background data
        for i in range(len(anchor_query)):
            if anchor_query not in background_set:
                anchor_query = shorten_query(anchor_query)
            else:
                # If match found, proceed as described in next-query prediction
                highest_ADJ_queries = ADJ_function(anchor_query)
                # target Query is the positive candidate if it is in the 20 queries,
                # the other 19 are negative candidates
                if target_query in highest_ADJ_queries:
                    # one_hot_session_vector[i] = 1
                    # then add the session to the train, val, test data

                    used_sessions.append(highest_ADJ_queries)
        break

    return used_sessions


df = read_files(1)

# begin = 0
# end = 5
parts = len(df['Query'])/10
sessions = np.array_split(df['Query'], parts)
sessions = sessions[:1000]

# for sess_nr in range(0, len(df)):
#     session = np.array(df['Query'][begin:end])
#     sessions[str(sess_nr)] = session
#     begin += 5
#     end += 5


options = sessions
data_next_query, corresponding_queries = next_query_prediction(sessions, options)
print(data_next_query)


#
## make list of all queries
# query_list = [session[:] for session in sessions]
# query_frequency_list = count_query_frequency(query_list)
# highest_100 = take_100_most_frequent(query_frequency_list)
# highest_100_prob = calculate_noise_prob(highest_100)
# for session in sessions:
#     # for each entry in the training, val and test set insert noisy query at random position
#     random_place = np.random.randint(0,len(session))
#     # probability of sampling a noisy query is proportional to frequency of query in background set
#     noise = get_random_noise(highest_100_prob)
#     session[random_place] = noise
# data_noisy = next_query_prediction(sessions)
# lambdaMart(data_noisy)

#
# data_long_tail = make_long_tail_set(sessions)
# lambdaMart(data_long_tail)

# What are we doing with queries who are the same as the previouse query?
