import logging
import math
import os
import pickle as pkl
import random
from collections import Counter
#from itertools import izip

import numpy as np
import pandas as pd
#from rankpy.models import LambdaMART
#from rankpy.queries import Queries

hred_use = False
training = True

import features.adj as ad
#import features.cossimilar as cs
#import features.length as lg
#import features.lengthdiff as ld
#import features.levenstein as levs
#if hred_use == True:
#    import features.HRED as hredf
import features.bg_count as bgcount

adj = ad.ADJ()
#lev = levs.Levenshtein()
#lendif = ld.LengthDiff()
#leng = lg.Length()
#coss = cs.CosineSimilarity()
#if hred_use == True:
#    hred = hredf.HRED()
bgc = bgcount.BgCount()


def get_query_index_pointers(dataset):
    query_index_pointers = []
    lower_bound = 0
    print("in get qip: " + str(dataset.shape[0]))
    for i in range(dataset.shape[0] / 20 + 1):
        query_index_pointer = lower_bound
        query_index_pointers.append(query_index_pointer)
        lower_bound += 20
        if i == dataset.shape[0] / 20:
            print(dataset.shape[0] / 20)
            print(i)
            print(lower_bound)
    query_index_pointers = np.array(query_index_pointers)
    return query_index_pointers


def lambdaMart(data, data_val, data_test, experiment_string):
    # Turn on logging.
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    query_index_pointers = get_query_index_pointers(data[:, 0])
    query_index_pointers_val = get_query_index_pointers(data_val[:, 0])
    print(query_index_pointers)
    query_index_pointers_test = get_query_index_pointers(data_test[:, 0])


    logging.info('================================================================================')

#     Set them to queries
    logging.info('Creating Training queries')

    training_targets = pd.DataFrame(data[:, :1]).astype(np.float32)
    print(training_targets.shape)
    training_features = pd.DataFrame(data[:, 1:]).astype(np.float32)
    print(training_features.shape)
    training_queries = Queries(training_features, training_targets, query_index_pointers)

    validation_targets = pd.DataFrame(data_val[:, :1]).astype(np.float32)
    print(validation_targets.shape)
    validation_features = pd.DataFrame(data_val[:, 1:]).astype(np.float32)
    print(validation_features.shape)
    validation_queries = Queries(validation_features, validation_targets, query_index_pointers_val)

    test_targets = pd.DataFrame(data_test[:, :1]).astype(np.float32)
    test_features = pd.DataFrame(data_test[:, 1:]).astype(np.float32)
    test_queries = Queries(test_features, test_targets, query_index_pointers_test)


    logging.info('================================================================================')

#     Print basic info about query datasets.
    logging.info('Train queries: %s' % training_queries)
    logging.info('Valid queries: %s' % validation_queries)
    logging.info('Test queries: %s' % test_queries)

    logging.info('================================================================================')

    model = LambdaMART(metric='nDCG@20', n_estimators=30) #, subsample=0.5)
    if training:
        model.fit(training_queries, validation_queries=validation_queries)
    else:
        model.load('../models/LambdaMART_next_query_HREDnDCG@20')
    logging.info("[Created model, starting training...]")


    logging.info('================================================================================')

    logging.info('%s on the test queries: %.8f'
                 % (model.metric, model.evaluate(test_queries, n_jobs=-1)))

    test_set_length = data_test.shape[0]
    logging.info('ADJ score')
    test_indices = test_set_length/20
    rankings = [np.arange(20) for _ in range(test_indices)]

    indexes_ones = [q % 20 for q, [x] in enumerate(data_test[:, :1]) if x == 1.0]
    mean_rank = mmr(indexes_ones, rankings)

    text_file = open("Results" + experiment_string + "sub.txt", "w")

    text_file.write("ADJ MRR: %s" % mean_rank)
    logging.info('MRR: %s' % mean_rank)

    logging.info('================================================================================')

    logging.info('Model score')
    scores = model.predict(test_queries)
    # print scores
    rankings, scores = model.predict_rankings(test_queries, return_scores=True)
    mean_rank = mmr(indexes_ones, rankings)

    text_file.write("LambdaMART MRR: %s" % mean_rank)
    text_file.write("nDCG@20 on lambdaMART: %s" % model.evaluate(test_queries, n_jobs=-1))

    logging.info('MRR: %s' % mean_rank)

    print('[Done training, saving model...]')
    model.save('../models/LambdaMART_' + experiment_string + model.metric)
    print("Done with experiment" + experiment_string)
    text_file.close()

def mmr(indexes, ranks):
    return np.mean([1 / (r[i] + 1.0) for i, r in izip(indexes, ranks)])



def create_features(anchor_query, session):
    lev_features = []
    lendif_features = []
    leng_features = []
    coss_features = []

    session_length = len(session)
    adj_dict = adj.adj_function(anchor_query)
    highest_adj_queries = adj_dict['adj_queries']
    sugg_features = adj_dict['absfreq']
    bgcount_features = bgc.calculate_feature(anchor_query, highest_adj_queries)
    if hred_use == True:
        hred_features = hred.calculate_feature(anchor_query, highest_adj_queries)
    for query in highest_adj_queries:
        if session_length > 11:
            # Take the features of the 10 most recent queries (contextual features)
            lev_features_per_query = lev.calculate_feature(query, session[-11:-1])
            lendif_per_query = lendif.calculate_feature(query, session[-11:-1])
            leng_per_query = leng.calculate_feature(query, session[-11:-1])
            coss_per_query = coss.calculate_feature(query, session[-11:-1])
        else:
            # If there are no 10 most recent queries: add zero padding at the end
            lev_features_per_query = lev.calculate_feature(query, session[:session_length - 1])
            lendif_per_query = lendif.calculate_feature(query, session[:session_length - 1])
            leng_per_query = leng.calculate_feature(query, session[:session_length - 1])
            coss_per_query = coss.calculate_feature(query, session[:session_length - 1])

            length_difference = 10 - (session_length - 1)
            zeros = [0 for _ in range(length_difference)]
            lev_features_per_query += zeros
            lendif_per_query += zeros
            leng_per_query += zeros
            coss_per_query += zeros

        lev_features.append(lev_features_per_query)
        lendif_features.append(lendif_per_query)
        leng_features.append(leng_per_query)
        coss_features.append(coss_per_query)
    features = np.vstack((np.array(sugg_features), np.array(bgcount_features)))
    features = np.vstack((features, np.transpose(np.array(lev_features))))
    features = np.vstack((features, np.transpose(np.array(lendif_features))))
    features = np.vstack((features, np.transpose(np.array(leng_features))))
    features = np.vstack((features, np.transpose(np.array(coss_features))))
    if hred_use == True:
        features = np.vstack((features, np.transpose(np.array(hred_features))))
    return features, highest_adj_queries


def create_dataframe_headers():
    headers = ["target", "suggestion", "bgcount"]
    lev_headers = ["levenshtein" + str(q) for q in range(10)]
    headers += lev_headers
    lendif_headers = ["lendif" + str(q) for q in range(10)]
    headers += lendif_headers
    length_headers = ["length" + str(q) for q in range(10)]
    headers += length_headers
    cossim_headers = ["cossim" + str(q) for q in range(10)]
    headers += cossim_headers
    if hred_use == True:
        headers.append("HRED")
    return headers



def next_query_prediction(sessions, experiment_string):
    used_sess = 0
    headers = create_dataframe_headers()
    for i, session in enumerate(sessions):
        anchor_query = session[-2]
        target_query = session[-1]
        # extract 20 queries with the highest ADJ score (most likely to follow the anchor query in the data)
        adj_dict = adj.adj_function(anchor_query)
        highest_adj_queries = adj_dict['adj_queries']
        features, highest_adj_queries = create_features(anchor_query, session)
        target_vector = np.zeros(len(highest_adj_queries))
        [target_query_index] = [q for q, x in enumerate(highest_adj_queries) if x == target_query]
        target_vector[target_query_index] = 1
        # then add the session to the train, val, test data
        sess_data = np.vstack((np.transpose(target_vector), features))
        if used_sess == 0:
            lambdamart_data = sess_data
            used_sess += 1
        else:
            lambdamart_data = np.hstack((lambdamart_data, sess_data))
            used_sess += 1
        if used_sess % 1000 == 0:
            print("[Visited %s anchor queries.]" % used_sess)
    lambda_dataframe = pd.DataFrame(data=np.transpose(lambdamart_data), columns=headers)
    lambda_dataframe.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
    print("---" * 30)
    print("used sessions:" + str(used_sess))

def shorten_query(query):
    query = query.rsplit(' ', 1)[0]
    return query


def make_long_tail_set(sessions, experiment_string):
    used_sess = 0
    headers = create_dataframe_headers()
    for session in sessions:
        # get anchor query and target query from session
        not_longtail = False
        anchor_query = session[-2]
        target_query = session[-1]
        # Cannot use ADJ
        # Therefore iteratively shorten anchor query by dropping terms
        # until we have a query that appears in the Background data
        for i,j in enumerate(range(len(anchor_query.split()))):
            [background_count] = bgc.calculate_feature(None, [anchor_query])
            if background_count == 0 and len(anchor_query.split()) == 1:
                not_longtail = True
                break
            if background_count == 0 and len(anchor_query.split()) > 1:
                print("shortened")
                anchor_query = shorten_query(anchor_query)
            else:
                if i > 0:
                    break
                else:
                    not_longtail = True
                    break
        if not_longtail:
            continue
        features, highest_adj_queries = create_features(anchor_query, session)
        if len(highest_adj_queries) < 20:
            continue
        # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
        target_vector = np.zeros(len(highest_adj_queries))
        index_list = [q for q, x in enumerate(highest_adj_queries) if x == target_query]
        if index_list == []:
            continue
        else:
            [target_query_index] = [q for q, x in enumerate(highest_adj_queries) if x == target_query]
        target_vector[target_query_index] = 1
        # then add the session to the train, val, test data
        sess_data = np.vstack((np.transpose(target_vector), features))
        if used_sess == 0:
            lambdamart_data = sess_data
            used_sess += 1
        else:
            lambdamart_data = np.hstack((lambdamart_data, sess_data))
            used_sess += 1
        if used_sess % 1000 == 0:
            print("[Visited %s anchor queries.]" % used_sess)
    lambda_dataframe = pd.DataFrame(data=np.transpose(lambdamart_data), columns=headers)
    lambda_dataframe.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
    print("---" * 30)
    print("used sessions:" + str(used_sess))

def count_query_frequency():
    background = [query for session in adj.bg_sessions for query in session]
    counts = Counter(background)
    highest_100 = counts.most_common(100)
    noise_freq = {q:f for q, f in highest_100}
    return noise_freq


def get_random_noise(noise_prob):
    norm = sum(noise_prob.values())
    probs = np.array(list(noise_prob.values()), np.float32) / norm
    # probability of sampling a noisy query is proportional to frequency of query in background set
    return np.random.choice(list(noise_prob.keys()), p=probs)

def noisy_query_prediction(sessions):
    print('[Creating noisy queries.]')
    noise_freq = count_query_frequency()
    for session in sessions:
        # for each entry in the training, val and test set insert noisy query at random position
        random_place = np.random.randint(0, len(session)-1)
        noise = get_random_noise(noise_freq)
        session.insert(random_place, noise)
    noisy_sessions = sessions
    return noisy_sessions


# Do LambdaMart for 3 different scenario's
# 1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:

#experiment_string = "next_query"
#print("Performing experiment: " + experiment_string)
#corresponding_queries = next_query_prediction(sessions, experiment_string)
#print("---" * 30)
#
## 2 RobustPrediction (when the context is perturbed with overly common queries)
## label 100 most frequent queries in the background set as noisy
#
#for i, session in enumerate(sessions):
#    if i == 0:
#        background_set = session
#    else:
#        background_set += session
#
#experiment_string = "noisy"
#print("Performing experiment: " + experiment_string)
#noisy_query_sessions = noisy_query_prediction(sessions, background_set)
#corresponding_queries_noisy = next_query_prediction(noisy_query_sessions, experiment_string)
#print("---" * 30)
#
## 3 Long-TailPrediction (when the anchor is not present in the background data)
## train, val and test set retain sessions for which the anchor query has not been
## seen in the background set (long-tail query)
#
#experiment_string = "long_tail"
#print("Performing experiment: " + experiment_string)
#corresponding_queries_lt = make_long_tail_set(sessions, experiment_string)
#print("---" * 30)
