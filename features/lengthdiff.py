from features.feature import Feature


class LengthDiff(Feature):

    @staticmethod
    def calculate_feature(compared_query, queries):
        features = []
        cql = len(compared_query)
        for q in queries:
            features.append(abs(len(q)-cql))
            Feature.cooccurrences[compared_query]['length'] = features
        return features
