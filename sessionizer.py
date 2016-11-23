'''
How to use this script:
1. Set the data path to the path where your preprocessed files are.
2. Run the script from the command line: python sessionizer.py -n 10  if you want to preprocess all files. Default is the first two files
'''


class Sessionizer():
    def __init__(self):
        pass

    data_path = "../data/tr_session.ctx"

    def get_sessions(self):
        sessions = []
        with open(self.data_path) as f:
            for line in f:
                sessions.append(line.rstrip('\n').split('\t'))
        return sessions


def __main__():
    sessionizer = Sessionizer()
    sessionizer.get_sessions()
    print 'got sessions'


__main__()
