from adj import ADJ


class Feature(ADJ):

    def score_query(self, anchor_query):
        if anchor_query in Feature.cooccurrences:
            queries = Feature.cooccurrences[anchor_query]['adj_queries']
        else:
            print "calculate queries"
            queries = Feature.adj_function(anchor_query)['adj_queries']

        return self.calculate_feature(anchor_query, queries)

    @staticmethod
    def calculate_feature(compared_query, queries):
        print "should be passed"
        pass
