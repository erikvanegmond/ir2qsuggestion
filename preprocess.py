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

data_path = "../AOL-user-ct-collection/"
output_data_path = "../preprocessed-data/"

word_counter = Counter()

def preprocess_files(max_files=10):
    print "reading {} files".format(max_files)
    file_counter = 0
    df = pd.DataFrame()
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            data = pd.read_csv(file_path, sep="\t", usecols=[0, 1, 2])
            print("preprocessing {}...".format(file_path))
            data = preprocess_data(data)
            save_df(output_data_path, data, fn)
            file_counter += 1


def alphanumeric_preprocessor(text):
    """
    Removes non alphanumeric characters and converts to lowercase and updates the word counter
    :param text:
    :return:
    """
    try:
        s = re.sub("\s\s+", " ", re.sub("[^a-z1-9]", " ", text.lower()))
        word_counter.update(s.split())
        return s
    except:
        # most likely nan, but if it fails just assume empty.
        return ""


def preprocess_data(df):
    """
    Preprocesses the data
    :param df:
    :return:
    """
    # take alphanumeric characters and make lowercase
    df.loc[:, 'Query'] = df['Query'].apply(alphanumeric_preprocessor)
    # TODO spell corrector
    return df


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfiles", type=int, default=2,
                        help="number of files to process")
    args = parser.parse_args()

    preprocess_files(args.nfiles)
__main__()
