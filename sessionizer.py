from itertools import izip


class Sessionizer(object):
    def __init__(self, data_path="../data/tr_session"):
        '''
        NOTE: the data path has no extention
        :param data_path:
        '''
        self.data_path = data_path
        self.sessions = []
        self.number_sessions = []

    def find_all_sessions(self):
        print "locating all sessions"
        self.sessions = []
        self.number_sessions = []
        with open(self.data_path + ".ctx") as ctx, open(self.data_path + ".out") as out:
            for ctx_line, out_line in izip(ctx, out):
                queries = ctx_line.rstrip('\n').split('\t')
                if len(set(queries)) > 1:
                    self.sessions.append(queries)
                    self.number_sessions.append([map(int, x.split()) for x in out_line.rstrip('\n').split('\t')])

    def get_sessions(self):
        if not len(self.sessions):
            self.find_all_sessions()

        return self.sessions

    def get_sessions_with_numbers(self):
        if not len(self.number_sessions):
            self.find_all_sessions()
        else:
            print "no need to get sessions again, we have them!"

        return self.number_sessions
