from adj import ADJ
from nltk.metrics.distance import edit_distance


class Levenshtein(ADJ):
    def score_query(self, anchor_query):
        if anchor_query in Levenshtein.cooccurrences:
            queries = Levenshtein.cooccurrences[anchor_query]['adj_queries']
        else:
            print "don't have it, fixing the queries somehow."
            queries = self.adj_function(anchor_query)['adj_queries']

        features = []
        for q in queries:
            features.append(self.lev_dist(anchor_query, q))
        Levenshtein.cooccurrences[anchor_query]['levenshtein'] = features
        return features

    @staticmethod
    def lev_dist(first, second):
        return edit_distance(first, second)
