from nltk.metrics.distance import edit_distance
from features.feature import Feature


class Levenshtein(Feature):

    @staticmethod
    def calculate_feature(compared_query, queries):
        features = []
        for q in queries:
            features.append(Levenshtein.lev_dist(compared_query, q))
        Feature.cooccurrences[compared_query]['levenshtein'] = features
        return features

    @staticmethod
    def lev_dist(first, second):
        return edit_distance(first, second)
