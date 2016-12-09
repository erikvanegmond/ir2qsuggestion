import csv
from _license import defaultdict
from collections import Counter
from datetime import *
import pandas as pd
import os
import argparse
import re
import pickle as pkl

'''
This script will generate 12 files. a .out .ctx and a .new file for each dataset {bg, train, test, validation}
.ctx and .out should be as you all know. .new are the potential new features.
Each line is session, queries seperated by tabs (\t), each query has multiple features separated by commas (,):
click_bool, click_rank, domain, time_since_last_query,time_since_last_click
Click rank is 0 if there is no click. domain, time_since_last_query,time_since_last_click are empty if not available.
'''
'''
TODO
* collect new features per session (average time between queries, number of queries, session lenght)
'''


class Preprocessor:
    data_path = "../AOL-user-ct-collection/"
    output_data_path = "../data/"

    word_counter = Counter()
    most_common_words = dict()
    datetime_mask = '%Y-%m-%d %H:%M:%S'

    def __init__(self, vocab_file_path="../data/aol_vocab.dict.pkl"):
        print "init"
        pkl_file = open(vocab_file_path, 'rb')
        vocab = pkl.load(pkl_file)
        self.w2n = defaultdict(int, {w: n1 for w, n1, n2 in vocab})
        self.n2w = {n1: w for w, n1, n2 in vocab}
        self.thirty_minutes = timedelta(minutes=30)
        self.time_zero = timedelta(minutes=0)
        print "loaded"

    def preprocess_files_first_pass(self, max_files=10):
        files = {'bg': {'ctx': open('{}bg_session.ctx'.format(self.output_data_path), 'w'),
                        'out': open('{}bg_session.out'.format(self.output_data_path), 'w'),
                        'new': open('{}bg_session.new'.format(self.output_data_path), 'w')},
                 'test': {'ctx': open('{}test_session.ctx'.format(self.output_data_path), 'w'),
                          'out': open('{}test_session.out'.format(self.output_data_path), 'w'),
                          'new': open('{}test_session.new'.format(self.output_data_path), 'w')},
                 'validation': {'ctx': open('{}val_session.ctx'.format(self.output_data_path), 'w'),
                                'out': open('{}val_session.out'.format(self.output_data_path), 'w'),
                                'new': open('{}val_session.new'.format(self.output_data_path), 'w')},
                 'train': {'ctx': open('{}tr_session.ctx'.format(self.output_data_path), 'w'),
                           'out': open('{}tr_session.out'.format(self.output_data_path), 'w'),
                           'new': open('{}tr_session.new'.format(self.output_data_path), 'w')}}

        print "reading {} files at {}".format(max_files, self.data_path)
        file_counter = 0
        for fn in os.listdir(self.data_path):
            file_path = self.data_path + fn
            if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
                with open(file_path, 'rb') as csv_file:
                    print "processing {}".format(file_path)
                    reader = csv.DictReader(csv_file, delimiter='\t')
                    last_time = 0
                    last_query_time = 0
                    delta_time = self.time_zero
                    queries = []
                    session_list_ctx = []
                    session_list_out = []
                    session_list_new = []
                    anonID = '0'
                    for i, row in enumerate(reader):
                        if i%10000 is 0:
                            print "{} lines processed".format(i)

                        cur_time = datetime.strptime(row['QueryTime'], self.datetime_mask)
                        if last_time:
                            delta_time = cur_time - last_time
                        else:
                            delta_time = self.time_zero

                        if delta_time > self.thirty_minutes or (anonID != row['AnonID'] and anonID != '0'):
                            self.write_data(files, session_list_out, session_list_ctx, session_list_new, queries)
                            last_time = 0
                            last_query_time = 0
                            delta_time = self.time_zero
                            session_list_ctx = []
                            session_list_out = []
                            session_list_new = []
                            queries = []

                        anonID = row['AnonID']
                        query = self.alphanumeric_preprocessor(row['Query'])
                        session_list_ctx.append(query)
                        session_list_out.append(self.text2out(query))
                        new_feature, last_query_time = self.new_features(row, cur_time, delta_time, last_query_time)
                        session_list_new.append(new_feature)
                        queries.append(row)
                        last_time = datetime.strptime(row['QueryTime'], self.datetime_mask)
                self.write_data(files, session_list_out, session_list_ctx, session_list_new, queries)

        self.close_files(files)

    @staticmethod
    def new_features(log_line, cur_time, delta_time, last_click_time):
        '''(click bool, click rank, click domain, time since last query, time since last click)'''
        click_bool = 0
        click_rank = 0
        time_since_last_click = ""
        time_since_last_query = ""
        if len(log_line['ItemRank']):
            click_bool = 1
            click_rank = log_line['ItemRank']
            last_click_time = cur_time
        domain = log_line['ClickURL']
        if last_click_time:
            time_since_last_click = (cur_time - last_click_time).total_seconds()

        time_since_last_query = delta_time.total_seconds()

        return "{}, {}, {}, {}, {}".format(click_bool, click_rank, domain, time_since_last_query,
                                           time_since_last_click), last_click_time

    def text2out(self, text):
        out = []
        for word in text.split():
            out.append(self.w2n[word])
        return " ".join(map(str, out))

    def write_data(self, files, session_list_out, session_list_ctx, session_list_new, queries):
        t = datetime.strptime(queries[0]['QueryTime'], self.datetime_mask)
        if t < datetime(year=2006, month=05, day=01):
            data_set = files['bg']
        elif t < datetime(year=2006, month=05, day=14):
            data_set = files['train']
        elif t < datetime(year=2006, month=05, day=21):
            data_set = files['test']
        else:
            data_set = files['validation']

        data_set['ctx'].write("\t".join(session_list_ctx) + "\n")
        data_set['out'].write("\t".join(session_list_out) + "\n")
        data_set['new'].write("\t".join(session_list_new) + "\n")

    @staticmethod
    def close_files(files):
        for s in files:
            for f in files[s]:
                files[s][f].close()

    def preprocess_files_second_pass(self, max_files=10):
        """
        In the second pass the infrequent words are removed and start and stop symbols added.
        :param max_files:
        :return:
        """
        print "reading {} files".format(max_files)
        file_counter = 0
        for fn in os.listdir(self.output_data_path):
            file_path = self.output_data_path + fn
            if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
                # print("loading {}...".format(file_path))
                # data = pd.read_csv(file_path, sep=",")
                # print("preprocessing {}...".format(file_path))
                # data = self.preprocess_data_second(data)
                # save_df(self.output_data_path, data, fn)
                # file_counter += 1
                return

    def alphanumeric_preprocessor(self, text):
        """
        Removes non alphanumeric characters and converts to lowercase and updates the word counter
        :param text:
        :return:
        """
        try:
            s = re.sub("\s\s+", " ", re.sub("[^a-z1-9]", " ", text.lower()))
            return s
        except:
            # most likely nan, but if it fails just assume empty.
            return ""

    def preprocess_data_first(self, df):
        """
        Preprocesses the data
        :param df:
        :return:
        """
        # take alphanumeric characters and make lowercase
        df.loc[:, 'Query'] = df['Query'].apply(self.alphanumeric_preprocessor)
        # TODO spell corrector
        return df

    def start_stop_commonizer(self, text):
        try:
            new_text = " ".join([x for x in text.split() if x in self.most_common_words])
            return "startsymbol {} stopsymbol".format(new_text)
        except:
            return "startsymbol stopsymbol"

    def preprocess_data_second(self, df):
        """
        Preprocesses the data
        :param df:
        :return:
        """
        # take alphanumeric characters and make lowercase
        df.loc[:, 'Query'] = df['Query'].apply(self.start_stop_commonizer)
        # TODO spell corrector
        return df


def __main__():
    pp = Preprocessor()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfiles", type=int, default=2,
                        help="number of files to process")
    parser.add_argument('--redo-wordcount', dest='redo_wordcount', action='store_true')

    args = parser.parse_args()

    print "go"
    pp.preprocess_files_first_pass(args.nfiles)

    # if pp.count_words:
    #     output = open('wordcounts.pkl', 'wb')
    #     print "keeping 90000 words from {}".format(len(pp.word_counter))
    #     pkl.dump(pp.word_counter.most_common(90000), output)
    #     output.close()
    #
    # pp.most_common_words = dict(pp.word_counter)
    #
    # pp.preprocess_files_second_pass(args.nfiles)


__main__()
