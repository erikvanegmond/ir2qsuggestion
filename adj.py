from collections import Counter
from collections import defaultdict

from sessionizer import Sessionizer
import numpy as np


class ADJ:
    def __init__(self, sessions_file="../data/test_session"):
        sessionizer = Sessionizer(data_path=sessions_file)
        self.sessions = sessionizer.get_sessions()
        self.cooccurrences = defaultdict(list)

    def adj_function(self, anchor_query, norm=True):
        if anchor_query in self.cooccurrences:
            return self.cooccurrences[anchor_query]

        cooccurrence_list = Counter()
        for session in self.sessions:
            if anchor_query in session:
                anchor_occurrence = [i for i, x in enumerate(session) if anchor_query == x]
                for i in anchor_occurrence:
                    if i < len(session) - 1:
                        cooccurrence_list.update([session[i + 1]])

        top20 = cooccurrence_list.most_common(20)
        top20_queries = [x for x, y in top20]

        if norm:
            tot = sum([y for x, y in top20])
            top20_freq = [y / float(tot) for x, y in top20]
        else:
            top20_freq = [y for x, y in top20]

        self.cooccurrences[anchor_query] = top20_queries, top20_freq
        return top20_queries, top20_freq
