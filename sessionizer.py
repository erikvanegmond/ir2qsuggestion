#from itertools import izip


class Sessionizer(object):
    def __init__(self, data_path="../data/tr_session"):
        '''
        NOTE: the data path has no extention
        :param data_path:
        '''
        self.data_path = data_path
        self.sessions = []
        self.number_sessions = []
        self.more_data = []
        self

    def find_all_sessions(self):
        print("locating sessions " + self.data_path)
        self.sessions = []
        self.number_sessions = []
        with open(self.data_path + ".ctx", 'r') as ctx, open(self.data_path + ".out", 'r') as out, open(self.data_path + ".new", 'r') as new_data:
            for ctx_line, out_line, new_line in zip(ctx, out, new_data):
                queries = ctx_line.rstrip('\n').split('\t')
                if len(set(queries)) > 1:
                    self.sessions.append(queries)
                    tmp = new_line.rstrip('\n').split('\t')
                    #For simplicity just returns the click bool and click rank. The other parts will need more guard codes. Reommend pre-processing with default values.
                    self.more_data.append([list(map(int, x.split(',')[:2])) for x in tmp])
                    self.number_sessions.append([list(map(int, x.split())) for x in out_line.rstrip('\n').split('\t')])

    def get_sessions(self):
        if not len(self.sessions):
            self.find_all_sessions()

        return self.sessions

    def get_sessions_with_numbers(self):
        if not len(self.number_sessions):
            self.find_all_sessions()

        return self.number_sessions

    def get_sessions_clickBool_clickRank(self):
        if not len(self.more_data):
            self.find_all_sessions()
        return self.more_data
        
        