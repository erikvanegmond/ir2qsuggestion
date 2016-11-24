from features.feature import Feature


class Length(Feature):

    @staticmethod
    def calculate_feature(compared_query, queries):
        print "cal len"
        features = []
        for q in queries:
            features.append(len(q))
            Feature.cooccurrences[compared_query]['length'] = features
        return features
