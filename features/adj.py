import os
from collections import defaultdict, Counter
from features.ranker import Ranker
import cPickle as pkl
from datetime import datetime
import numpy as np

class ADJ(Ranker):
    suitable_sessions = []
    bg_info = defaultdict(Counter)

    def __init__(self):
        super(ADJ, self).__init__()
        if not ADJ.bg_info:
            info_path = '../data/bg_info.pkl'
            if os.path.isfile(info_path):
                time = datetime.now().strftime('%d-%m %H:%M:%S')
                print("[%s: Loading bg_info.pkl...]" % time)
                with open(info_path, 'rb') as pkl_file:
                    ADJ.bg_info = pkl.load(pkl_file)
                time = datetime.now().strftime('%d-%m %H:%M:%S')
                print("[%s: Loaded %d sessions...]" % (time, len(ADJ.bg_info)))
            else:
                time = datetime.now().strftime('%d-%m %H:%M:%S')
                print("[%s: Creating co-occurrence info...]" % time)
                s = 0
                q = 0
                for session in ADJ.bg_sessions:
                    # Every query should add +1 for the other queries
                    for anchor_query in set(session):
                        q += 1
                        anchor_occurrence = [i for i, x in enumerate(session) if anchor_query == x]
                        for i in anchor_occurrence:
                            if i < len(session) - 1:
                                ADJ.bg_info[anchor_query].update([session[i + 1]])
                    s += 1
                    if s % 10000 == 0:
                        print("[%s sessions, %d queries]" % (s, q))
                time = datetime.now().strftime('%d-%m %H:%M:%S')
                print("[%s: Saving %d sessions...]" % (time, len(ADJ.bg_info)))
                with open(info_path, 'wb') as pkl_file:
                    pkl.dump(ADJ.bg_info, pkl_file)
        else:
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print("[%s: Background info is already loaded.]" % time)

    @staticmethod
    def adj_function(anchor_query):
        if anchor_query in ADJ.cooccurrences and 'adj_queries' in ADJ.cooccurrences[anchor_query]:
            return ADJ.cooccurrences[anchor_query]

        top20 = ADJ.bg_info[anchor_query].most_common(20)
        top20_queries = [x for x, y in top20]

        tot = sum([y for x, y in top20])
        top20_relfreq = [y / float(tot) for x, y in top20]
        top20_absfreq = [y for x, y in top20]

        ADJ.cooccurrences[anchor_query].update({'adj_queries': top20_queries, 'absfreq': top20_absfreq,
                                                'relfreq': top20_relfreq})
        return ADJ.cooccurrences[anchor_query]

    @staticmethod
    def find_suitable_sessions():
        suitable_sessions_fname = "../data/lm_test_sessions.pkl"
        if os.path.isfile(suitable_sessions_fname):
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print("[%s: Loading lm_val_sessions.pkl...]" % time)
            with open(suitable_sessions_fname, 'rb') as pkl_file:
                ADJ.suitable_sessions = pkl.load(pkl_file)
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print("[%s: Loaded %d sessions...]" % (time, len(ADJ.suitable_sessions)))
            return ADJ.suitable_sessions

        time = datetime.now().strftime('%d-%m %H:%M:%S')
        ADJ.suitable_sessions = []
        l = float(len(ADJ.sessions))
        print("[%s: Searching for sessions in %d sessions...]" % (time, l))
        for i, session in enumerate(ADJ.sessions):
            anchor_query = session[-2]
            target_query = session[-1]
            adj_dict = ADJ.adj_function(anchor_query)
            highest_adj_queries = adj_dict['adj_queries']
            if target_query in highest_adj_queries and 19 < len(highest_adj_queries):
                ADJ.suitable_sessions.append(session)

            if i % 120 == 0:
                print 'checked session {}, at {}%'.format(i, np.round(i / l * 100))
        time = datetime.now().strftime('%d-%m %H:%M:%S')
        print("[%s: Saving file lm_test_sessions.pkl...]" % time)
        pkl_file = open(suitable_sessions_fname, 'wb')
        pkl.dump(ADJ.suitable_sessions, pkl_file)
        pkl_file.close()
        return ADJ.suitable_sessions
