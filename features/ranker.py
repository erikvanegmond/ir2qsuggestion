from collections import defaultdict, Counter

import itertools

from sessionizer import Sessionizer
import pickle as pkl


class Ranker(object):
    sessionizer = None
    bg_sessionizer = None
    sessions = []
    bg_sessions = []
    cooccurrences = defaultdict(dict)
    w2n = None
    n2w = None
    query_counts = None

    def __init__(self, train_sessions_file="../data/tr_session", bg_sessions_file="../data/bg_session", vocab_file="../data/aol_vocab.dict.pkl"):
        if not len(Ranker.sessions):
            Ranker.sessionizer = Sessionizer(data_path=train_sessions_file)
            Ranker.sessions = Ranker.sessionizer.get_sessions()
            Ranker.bg_sessionizer = Sessionizer(data_path=bg_sessions_file)
            Ranker.bg_sessions = Ranker.bg_sessionizer.get_sessions()
            Ranker.query_counts = Counter(list(itertools.chain.from_iterable(Ranker.bg_sessions)))
        else:
            print "don't have to get the sessions as we already have them"

        if not Ranker.w2n:
            pkl_file = open(vocab_file, 'rb')
            vocab = pkl.load(pkl_file)
            Ranker.w2n = defaultdict(int, {w: n1 for w, n1, n2 in vocab})
            Ranker.n2w = {n1: w for w, n1, n2 in vocab}
