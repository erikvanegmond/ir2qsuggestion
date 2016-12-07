import cPickle as pickle
from model.model import Model
from features.feature import Feature
import utils


class DatasetFeature(Feature):
    features = {}
    model = None
    unsaved_changes = False

    def __init__(self, data_file="../data/HRED_features.pkl", model_file='../models/29-11_4.589_0_90005x1000x90005.npz'):
        if not len(DatasetFeature.features):
            DatasetFeature.features = pickle.load(open(data_file, 'rb'))
        else:
            print "Features already loaded."
            
        if not DatasetFeature.model:
            DatasetFeature.model = Model.load(model_file)
        else:
            print "Model already loaded."
            
    def save(self, data_file="../data/HRED_features.pkl"):
        if DatasetFeature.unsaved_changes:
            pickle.dump(DatasetFeature.features, open(data_file, 'wb'))

class HRED(DatasetFeature):
    @staticmethod
    def calculate_feature(compared_query, queries):
        fts = []
        for q in queries:
            if compared_query in DatasetFeature.features.keys():
                if q in DatasetFeature.features[compared_query].keys():
                    fts.append(DatasetFeature.features[compared_query][q])
                else:
                    print('Unknown suggestion: ' + q)
                    likelihood = HRED.add_likelihood(compared_query, q)
                    DatasetFeature.features[compared_query][q] = likelihood
                    fts.append(DatasetFeature.features[compared_query][q])
            else:
                print('Unknown anchor query: ' + compared_query)
                DatasetFeature.features[compared_query] = {}
                likelihood = HRED.add_likelihood(compared_query, q)
                DatasetFeature.features[compared_query][q] = likelihood
                fts.append(DatasetFeature.features[compared_query][q])
                
        HRED.cooccurrences[compared_query]['HRED'] = fts
        return fts
    @staticmethod
    def add_likelihood(compared_query, q):
        num_anchor_query = utils.vectorify(compared_query)
        num_sug_query = utils.vectorify(q)                    
        likelihood = DatasetFeature.model.likelihood(num_anchor_query, num_sug_query)
        DatasetFeature.unsaved_changes = True

        return likelihood
