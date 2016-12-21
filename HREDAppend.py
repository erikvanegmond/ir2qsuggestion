# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:53:11 2016

@author: morph
"""

#import logging
#import math
#import os
#import pickle as pkl
#import random
#from collections import Counter
#from itertools import izip

#import numpy as np
import pandas as pd
#from rankpy.models import LambdaMART
#from rankpy.queries import Queries

hred_use = True
training = True

import features.adj as ad
import features.cossimilar as cs
import features.length as lg
import features.lengthdiff as ld
import features.levenstein as levs
if hred_use == True:
    import features.HRED as hredf
import features.bg_count as bgcount

adj = ad.ADJ()
lev = levs.Levenshtein()
lendif = ld.LengthDiff()
leng = lg.Length()
coss = cs.CosineSimilarity()
if hred_use == True:
    hred = hredf.HRED()
bgc = bgcount.BgCount()


def getHRED_features(anchor_query):
    adj_dict = adj.adj_function(anchor_query)
    highest_adj_queries = adj_dict['adj_queries']
    hred_features = hred.calculate_feature(anchor_query, highest_adj_queries)
    return hred_features    

def next_query_hred_prediction(sessions, experiment_string, csv):
    used_sess = 0
    dataf = pd.read_csv(csv)
    for i, session in enumerate(sessions):
        anchor_query = session[-2]
        HREDFeatures = getHRED_features(anchor_query)
        dataf[i]['HRED'] = HREDFeatures
        used_sess += 1
        if used_sess % 1000 == 0:
            print("[Visited %s anchor queries.]" % used_sess)
    dataf.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
    print("---" * 30)
    print("used sessions:" + str(used_sess))

def make_long_tail_hred_set(sessions, experiment_string, csv):
    used_sess = 0
    dataf = pd.read_csv(csv)
    for session in sessions:
        not_longtail = False
        anchor_query = session[-2]
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
        HREDFeatures = getHRED_features(anchor_query)
        dataf[i]['HRED'] = HREDFeatures
        used_sess += 1
        if used_sess % 1000 == 0:
            print("[Visited %s anchor queries.]" % used_sess)
    dataf.to_csv('../data/lamdamart_data_' + experiment_string + '.csv')
    print("---" * 30)
    print("used sessions:" + str(used_sess))

    
def shorten_query(query):
    query = query.rsplit(' ', 1)[0]
    return query
