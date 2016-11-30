from collections import Counter
from collections import defaultdict

from ranker import Ranker
from sessionizer import Sessionizer
import numpy as np


class ADJ(Ranker):
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
