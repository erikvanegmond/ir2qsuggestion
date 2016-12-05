import logging
import math
import os
import pickle as pkl
import random
from collections import Counter
from itertools import izip

import numpy as np
import pandas as pd
from rankpy.models import LambdaMART
from rankpy.queries import Queries

hred_use = True

import features.adj as ad
import features.cossimilar as cs
import features.length as lg
import features.lengthdiff as ld
import features.levenstein as levs
if hred_use == True:
    import features.HRED as hredf
import features.bg_count as bgcount

pkl_file = open('../data/lm_tr_sessions.pkl', 'rb')
sessions = pkl.load(pkl_file)
print(sessions[1])
pkl_file.close()

adj = ad.ADJ()
adj.find_suitable_sessions()
lev = levs.Levenshtein()
lendif = ld.LengthDiff()
leng = lg.Length()
coss = cs.CosineSimilarity()
if hred_use == True:
    hred = hredf.HRED()
bgc = bgcount.BgCount()


def get_query_index_pointers(dataset):
    query_index_pointers = []
    lower_bound = 0
    for i in range(dataset.shape[0] / 20 + 1):
        query_index_pointer = lower_bound
        query_index_pointers.append(query_index_pointer)
        lower_bound += 20
    query_index_pointers = np.array(query_index_pointers)
    return query_index_pointers


def lambdaMart(data, experiment_string):
    print("done with experiment" + experiment_string)

    # Turn on logging.
    #logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    # divide set into train, val and test set
    # 55% train
    # 20% validation
    # 25% test
    #query_index_pointers = get_query_index_pointers(data[:, 0])

    #train_part_pointer = int(math.floor(query_index_pointers.shape[0] * 0.55))
    #training_pointers, test_val_pointers = query_index_pointers[:train_part_pointer], query_index_pointers[
    #                                                                                  train_part_pointer - 1:]
    #val_part = int(math.floor(test_val_pointers.shape[0] * 0.40))
    #training_length = training_pointers.shape[0]
    #upper_bound_train = training_pointers[training_length - 1]
    #validation_pointers = test_val_pointers[:val_part]
    #validation_length = validation_pointers.shape[0]
    #upper_bound_val = validation_pointers[validation_length - 1]
    #test_pointers = test_val_pointers[val_part - 1:]

    #training_queries = data[:upper_bound_train, :]

    #validation_queries, test_queries = data[upper_bound_train:upper_bound_val, :], data[upper_bound_val:, :]

    #logging.info('================================================================================')

    # Set them to queries
    #logging.info('Creating Training queries')
    #training_targets = pd.DataFrame(training_queries[:, :1]).astype(np.float32)
    #training_features = pd.DataFrame(training_queries[:, 1:]).astype(np.float32)
    #training_queries = Queries(training_features, training_targets, training_pointers)

    #logging.info('Creating Validation queries')
    #validation_targets = pd.DataFrame(validation_queries[:, :1]).astype(np.float32)
    #validation_features = pd.DataFrame(validation_queries[:, 1:]).astype(np.float32)
    #validation_pointers = validation_pointers - upper_bound_train
    #validation_queries = Queries(validation_features, validation_targets, validation_pointers)

    #logging.info('Creating Test queries')
    #test_targets = pd.DataFrame(test_queries[:, :1]).astype(np.float32)
    #test_features = pd.DataFrame(test_queries[:, 1:]).astype(np.float32)
    #test_pointers = test_pointers - (upper_bound_val)
    #test_queries = Queries(test_features, test_targets, test_pointers)

    #logging.info('================================================================================')

    # Print basic info about query datasets.
    #logging.info('Train queries: %s' % training_queries)
    #logging.info('Valid queries: %s' % validation_queries)
    #logging.info('Test queries: %s' % test_queries)

    #logging.info('================================================================================')

    #model = LambdaMART(metric='nDCG@20', n_estimators=500, subsample=0.5)
    #logging.info("model is made")
    #model.fit(training_queries, validation_queries=validation_queries)

    #logging.info('================================================================================')

    #logging.info('%s on the test queries: %.8f'
     #            % (model.metric, model.evaluate(test_queries, n_jobs=-1)))

    #test_set = data[upper_bound_val:, :]
    #test_set_length = test_set.shape[0]
    #test_set_targets = test_set[:, :1]
    #logging.info('ADJ score')
    #test_indices = test_set_length/20
    #rankings = [np.arange(20) for i in range(test_indices)]


    #indexes_ones = [q % 20 for q, [x] in enumerate(test_set_targets) if x == 1.0]
    #mean_rank = np.mean([1/(r[i]+1) for i,r in izip(indexes_ones, rankings)])
    #logging.info('MRR: %s' % mean_rank)

    #logging.info('================================================================================')

    #logging.info('Model score')
    #rankings = model.predict_rankings(test_queries)
    #mean_rank = np.mean([1 / (r[i] + 1) for i, r in izip(indexes_ones, rankings)])
    #logging.info('MRR: %s' % mean_rank)

    #model.save('LambdaMART_L7_S0.1_E50_' + experiment_string + model.metric)


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
    bad_sess = 0
    corresponding_queries = []
    if os.path.isfile('lamdamart_data_next_query.csv'):
        print "read csv!"
        df = pd.read_csv('lamdamart_data_next_query.csv')
        df.drop('Unnamed: 0', 1)
        lambdamart_data = df.get_values()[:, 1:]
        print "loaded!!!!"
        # lambdaMart(lambdamart_data, experiment_string)
    else:
        headers = create_dataframe_headers()
        for i, session in enumerate(sessions):
            anchor_query = session[-2]
            target_query = session[-1]
            # extract 20 queries with the highest ADJ score (most likely to follow the anchor query in the data)
            adj_dict = adj.adj_function(anchor_query)
            highest_adj_queries = adj_dict['adj_queries']             # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
            if target_query in highest_adj_queries and 19 < len(highest_adj_queries):
                features, highest_adj_queries = create_features(anchor_query, session)
                print("Session: " + str(i))
                target_vector = -1 * np.ones(len(highest_adj_queries))
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
            else:
                bad_sess += 1
                continue
            if used_sess % 100 == 0:
                lambda_dataframe = pd.DataFrame(data=np.transpose(lambdamart_data), columns=headers)
                print("[Visited %s anchor queries. %d sessions were skipped.]" % (used_sess, bad_sess))
                lambda_dataframe.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
        lambdamart_data = np.transpose(lambdamart_data)

    # lambdaMart(np.transpose(lambdamart_data), experiment_string)
    print("---" * 30)
    print("used sessions:" + str(used_sess))
    return corresponding_queries


def shorten_query(query):
    query = query.rsplit(' ', 1)[0]
    return query


def make_long_tail_set(sessions, experiment_string):
    used_sess = 0
    bad_sess = 0
    corresponding_queries = []
    if os.path.isfile('../data/lamdamart_data_long_tail.csv'):
        print "read csv!"
        df = pd.read_csv('../data/lamdamart_data_long_tail.csv')
        df.drop('Unnamed: 0', 1)
        lambdamart_data = df.get_values()[:, 1:]
        print "loaded!!!!"
    else:
        headers = create_dataframe_headers()
        for i, session in enumerate(sessions):
            print("Session: " + str(i))
            session_length = len(session)
            # get anchor query and target query from session
            anchor_query = session[session_length - 2]
            target_query = session[session_length - 1]
            # Cannot use ADJ
            # Therefore iteratively shorten anchor query by dropping terms
            # until we have a query that appears in the Background data
            for j in range(len(anchor_query.split())):
                print(j)
                print(len(anchor_query.split()))
                [background_count] = bgc.calculate_feature(None, [anchor_query])
                print(background_count)
                if background_count == 0 and len(anchor_query.split()) != 1:
                    print("shortened")
                    anchor_query = shorten_query(anchor_query)
                else:
                    break
            features, highest_adj_queries = create_features(anchor_query, session)
            # target Query is the positive candidate if it is in the 20 queries, the other 19 are negative candidates
            if target_query in highest_adj_queries and 19 < len(highest_adj_queries):
                print("Session: " + str(i))
                target_vector = -1 * np.ones(len(highest_adj_queries))
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
            else:
                bad_sess += 1
                continue
            if hred_use == True:
                if used_sess == len(hred.features):
                    break
            if used_sess % 1000 == 0:
                lambda_dataframe = pd.DataFrame(data=np.transpose(lambdamart_data), columns=headers)
                print("[Visited %s anchor queries. %d sessions were skipped.]" % (used_sess, bad_sess))
                lambda_dataframe.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
        lambdamart_data = np.transpose(lambdamart_data)
    pd.read_csv('../data/lamdamart_data_long_tail.csv')
    print("---" * 30)
    print("used sessions:" + str(used_sess))
    # lambdaMart(np.transpose(lambdamart_data), experiment_string)
    return corresponding_queries


def count_query_frequency(query_list):
    noise_freq = []
    counts = Counter(np.array(query_list))
    highest_100 = counts.most_common(100)
    for i in range(100):
        noise_freq.append(highest_100[i][1])
    return highest_100, noise_freq


def get_random_noise(highest_100, noise_prob):
    total = sum(w for w in noise_prob)
    r = random.uniform(0, total)
    upto = 0
    for index, w in enumerate(noise_prob):
        if upto + w >= r:
            return highest_100[index][0]
        upto += w
    assert False, "Shouldn't get here"


def noisy_query_prediction(sessions, background_set):
    highest_100, noise_freq = count_query_frequency(background_set)
    for session in sessions:
        # for each entry in the training, val and test set insert noisy query at random position
        random_place = np.random.randint(0, len(session))
        noise = get_random_noise(highest_100, noise_freq)
        # probability of sampling a noisy query is proportional to frequency of query in background set
        session[random_place] = noise
    noisy_sessions = sessions
    return noisy_sessions


# Do LambdaMart for 3 different scenario's
# 1 Next-QueryPrediction (when anchor query exists in background data)
# for each session:

experiment_string = "next_query"
print("Performing experiment: " + experiment_string)
corresponding_queries = next_query_prediction(sessions, experiment_string)
print("---" * 30)

# 2 RobustPrediction (when the context is perturbed with overly common queries)
# label 100 most frequent queries in the background set as noisy

for i, session in enumerate(sessions):
    if i == 0:
        background_set = session
    background_set += session

experiment_string = "noisy"
print("Performing experiment: " + experiment_string)
noisy_query_sessions = noisy_query_prediction(sessions, background_set)
corresponding_queries_noisy = next_query_prediction(noisy_query_sessions, experiment_string)
print("---" * 30)

# 3 Long-TailPrediction (when the anchor is not present in the background data)
# train, val and test set retain sessions for which the anchor query has not been
# seen in the background set (long-tail query)

experiment_string = "long_tail"
print("Performing experiment: " + experiment_string)
corresponding_queries_lt = make_long_tail_set(sessions, experiment_string)
print("---" * 30)
