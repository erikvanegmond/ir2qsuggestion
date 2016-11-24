from nltk.metrics.distance import edit_distance

from features.feature import Feature


class Levenshtein(Feature):

    def calculate_feature(self, anchor_query, queries):
        features = []
        for q in queries:
            features.append(self.lev_dist(anchor_query, q))
            Feature.cooccurrences[anchor_query]['levenshtein'] = features
        return features

    @staticmethod
    def lev_dist(first, second):
        return edit_distance(first, second)
