import os
from collections import Counter
from ranker import Ranker
import cPickle as pkl

suitable_sessions_fname = "suitable_sessions.pkl"


class ADJ(Ranker):
    suitable_sessions = []

    @staticmethod
    def adj_function(anchor_query):
        if anchor_query in ADJ.cooccurrences and 'adj_queries' in ADJ.cooccurrences[anchor_query]:
            return ADJ.cooccurrences[anchor_query]

        cooccurrence_list = Counter()
        for session in ADJ.bg_sessions:
            if anchor_query in session:
                anchor_occurrence = [i for i, x in enumerate(session) if anchor_query == x]
                for i in anchor_occurrence:
                    if i < len(session) - 1:
                        cooccurrence_list.update([session[i + 1]])

        top20 = cooccurrence_list.most_common(20)
        top20_queries = [x for x, y in top20]

        tot = sum([y for x, y in top20])
        top20_relfreq = [y / float(tot) for x, y in top20]
        top20_absfreq = [y for x, y in top20]

        ADJ.cooccurrences[anchor_query].update({'adj_queries': top20_queries, 'absfreq': top20_absfreq,
                                                'relfreq': top20_relfreq})
        return ADJ.cooccurrences[anchor_query]

    @staticmethod
    def find_suitable_sessions():
        print "finding"
        if os.path.isfile(suitable_sessions_fname):
            pkl_file = open(suitable_sessions_fname, 'rb')
            ADJ.suitable_sessions = pkl.load(pkl_file)
            pkl_file.close()
            return ADJ.suitable_sessions

        ADJ.suitable_sessions = []
        l = float(len(ADJ.sessions))
        for i, session in enumerate(ADJ.sessions):
            anchor_query = session[-2]
            target_query = session[-1]
            adj_dict = ADJ.adj_function(anchor_query)
            highest_adj_queries = adj_dict['adj_queries']
            if target_query in highest_adj_queries and 19 < len(highest_adj_queries):
                ADJ.suitable_sessions.append(session)

            if i % 120 == 0:
                print 'checked session {}, at {}%'.format(i, i / l)

        pkl_file = open(suitable_sessions_fname, 'wb')
        pkl.dump(ADJ.suitable_sessions, pkl_file)
        pkl_file.close()
        return ADJ.suitable_sessions
