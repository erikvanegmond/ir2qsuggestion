from adj import ADJ


class Feature(ADJ):
    def score_query(self, anchor_query):
        if anchor_query in Feature.cooccurrences:
            queries = Feature.cooccurrences[anchor_query]['adj_queries']
        else:
            queries = self.adj_function(anchor_query)['adj_queries']

        self.calculate_feature(anchor_query, queries)

    def calculate_feature(self, anchor_query, queries):
        pass
