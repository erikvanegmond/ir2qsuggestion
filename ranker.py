from collections import defaultdict
from sessionizer import Sessionizer


class Ranker(object):
    sessions = []
    cooccurrences = defaultdict(dict)


    def __init__(self, sessions_file="../data/test_session"):
        if not len(Ranker.sessions):
            sessionizer = Sessionizer(data_path=sessions_file)
            Ranker.sessions = sessionizer.get_sessions()
        else:
            print "don't have to get the sessions as we already have them"