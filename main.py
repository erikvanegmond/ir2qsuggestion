import pandas as pd
import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer

data_path = "../sessionized-data/"

'''
    Sessions end after 30 minutes of inactivity, same definition google uses
    https://support.google.com/analytics/answer/2731565?hl=en
'''
session_inactivity = 30


def read_files(max_files=10):  # -> pd.DataFrame:
    print "reading {} files".format(max_files)
    file_counter = 0
    df = pd.DataFrame()
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            df = pd.concat(
                [df, pd.read_csv(file_path, sep=",", parse_dates=[2], infer_datetime_format=True)])
            file_counter += 1
    sessions = df.groupby(['AnonID', 'session_label'])
    for s in sessions:
        print s[1]
    return df


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfiles", type=int, default=1,
                        help="number of files to process")
    args = parser.parse_args()

    data = read_files(args.nfiles)
    # vectorizer = CountVectorizer(analyzer="word", \
    #                              tokenizer=None, \
    #                              preprocessor=None, \
    #                              stop_words=None, \
    #                              max_features=5000)
    # print "vectorizing..."
    # print vectorizer.fit_transform(data["Query"].values)

__main__()
