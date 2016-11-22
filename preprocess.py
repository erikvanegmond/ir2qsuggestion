from collections import Counter

import pandas as pd
import os
import argparse
import re
from utils import save_df
import pickle as pkl

'''
How to use this script:
1. Set the data path and the output datapath for your system. The output directory does not have to exist yet.
2. Run the script from the command line: python preprocess.py -n 10  if you want to preprocess all files.
'''


class Preprocessor:
    data_path = "../AOL-user-ct-collection/"
    output_data_path = "../preprocessed-data/"

    word_counter = Counter()
    count_words = True
    most_common_words = dict()

    def __init__(self):
        pass

    def preprocess_files_first_pass(self, max_files=10):
        print "reading {} files".format(max_files)
        file_counter = 0
        for fn in os.listdir(self.data_path):
            file_path = self.data_path + fn
            if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
                print("loading {}...".format(file_path))
                data = pd.read_csv(file_path, sep="\t", usecols=[0, 1, 2])
                print("preprocessing {}...".format(file_path))
                data = self.preprocess_data_first(data)
                save_df(self.output_data_path, data, fn)
                file_counter += 1

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
                print("loading {}...".format(file_path))
                data = pd.read_csv(file_path, sep=",")
                print("preprocessing {}...".format(file_path))
                data = self.preprocess_data_second(data)
                save_df(self.output_data_path, data, fn)
                file_counter += 1

    def alphanumeric_preprocessor(self, text):
        """
        Removes non alphanumeric characters and converts to lowercase and updates the word counter
        :param text:
        :return:
        """
        try:
            s = re.sub("\s\s+", " ", re.sub("[^a-z1-9]", " ", text.lower()))
            if self.count_words:
                self.word_counter.update(s.split())
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
    if os.path.isfile('wordcounts.pkl') and not args.redo_wordcount:
        pp.count_words = False
        pkl_file = open('wordcounts.pkl', 'rb')
        pp.word_counter = pkl.load(pkl_file)
    print "Counting words: {}".format(pp.count_words)

    pp.preprocess_files_first_pass(args.nfiles)

    if pp.count_words:
        output = open('wordcounts.pkl', 'wb')
        print "keeping 90000 words from {}".format(len(pp.word_counter))
        pkl.dump(pp.word_counter.most_common(90000), output)
        output.close()

    pp.most_common_words = dict(pp.word_counter)

    pp.preprocess_files_second_pass(args.nfiles)


__main__()
