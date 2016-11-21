import pandas as pd
import os
import argparse
import re
from utils import save_df
from sklearn.feature_extraction.text import CountVectorizer

'''
How to use this script:
1. Set the data path to the path where your preprocessed files are.
2. Run the script from the command line: python sessionizer.py -n 10  if you want to preprocess all files. Default is the first two files
'''

data_path = "../preprocessed-data/"
output_data_path = "../sessionized-data/"

'''
    Sessions end after 30 minutes of inactivity, same definition google uses
    https://support.google.com/analytics/answer/2731565?hl=en
'''
session_inactivity = 30


def read_files(max_files=10):  # -> pd.DataFrame:
    print "reading {} files".format(max_files)
    file_counter = 0
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            df = pd.read_csv(file_path, sep=",", parse_dates=[2], infer_datetime_format=True)
            print("preprocessing {}...".format(file_path))
            sessions(df)
            save_df(output_data_path, df, fn)
            file_counter += 1


def users_from_data(df):
    return df.groupby('AnonID')


def sessions(df):
    """
    :param df: dataframe to retrieve sessions from
    :return: labeled all rows to a with a session number
    """
    users = users_from_data(df)
    new_df = pd.DataFrame()
    for user in users:
        new_df = pd.concat([new_df, get_sessions_from_user(user[1])])
    return new_df


def get_sessions_from_user(user):
    # suppress annoying warning
    user.is_copy = False

    user.loc[:, 'delta_time'] = user['QueryTime'].diff().fillna(0)
    time_labels = []
    current_label = 0
    for i, row in user.iterrows():
        if not current_label:
            current_label = 1
        else:
            if row['delta_time'] > pd.Timedelta(minutes=30):
                current_label += 1
        time_labels.append(current_label)
    user.loc[:, 'session_label'] = pd.Series(time_labels, index=user.index)
    return user


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfiles", type=int, default=2,
                        help="number of files to process")
    args = parser.parse_args()
    read_files(args.nfiles)


__main__()
