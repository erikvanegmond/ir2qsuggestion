from features.feature import Feature


class BgCount(Feature):

    @staticmethod
    def calculate_feature(compared_query, queries):
        features = []
        for q in queries:
            features.append(BgCount.query_counts[q])
            Feature.cooccurrences[compared_query]['length'] = features
        return features
