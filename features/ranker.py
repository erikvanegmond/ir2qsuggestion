from collections import defaultdict
from sessionizer import Sessionizer
import pickle as pkl


class Ranker(object):
    sessions = []
    cooccurrences = defaultdict(dict)
    w2n = None
    n2w = None

    def __init__(self, sessions_file="../data/test_session", vocab_file="../data/aol_vocab.dict.pkl"):
        if not len(Ranker.sessions):
            sessionizer = Sessionizer(data_path=sessions_file)
            Ranker.sessions = sessionizer.get_sessions()
        else:
            print "don't have to get the sessions as we already have them"

        if not Ranker.w2n:
            pkl_file = open(vocab_file, 'rb')
            vocab = pkl.load(pkl_file)
            Ranker.w2n = defaultdict(int,{w: n1 for w, n1, n2 in vocab})
            Ranker.n2w = {n1: w for w, n1, n2 in vocab}
