import cPickle as pickle
from features.feature import Feature

class DatasetFeature(Feature):
    features = {}

    def __init__(self, data_file="../data/HRED_features.pkl"):
        if not len(DatasetFeature.features):
            DatasetFeature.features = pickle.load(open(data_file, 'rb'))
        else:
            print "Features already loaded."

class HRED(DatasetFeature):

    @staticmethod
    def calculate_feature(compared_query, queries):
        fts = []
        for q in queries:
            fts.append(DatasetFeature.features[compared_query][q])
            Feature.cooccurrences[compared_query]['HRED'] = DatasetFeature.features[compared_query][q]
        return fts