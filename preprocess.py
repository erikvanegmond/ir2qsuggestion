import pandas as pd
import os
import argparse
import re

'''
How to use this script:
1. Set the data path and the output datapath for your system. The output directory does not have to exist yet.
2. Run the script from the command line: python preprocess.py -n 10  if you want to preprocess all files.
'''

data_path = "../AOL-user-ct-collection/"
output_dat_path = "../preprocessed-data/"

'''
    Sessions end after 30 minutes of inactivity, same definition google uses
    https://support.google.com/analytics/answer/2731565?hl=en
'''
session_inactivity = 30


def save_df(df, fn):
    file_path = output_dat_path + fn
    if not os.path.exists(output_dat_path):
        os.makedirs(output_dat_path)
    df.to_csv(file_path)
    print "{} saved".format(file_path)


def preprocess_files(max_files=10):  # -> pd.DataFrame:
    print "reading {} files".format(max_files)
    file_counter = 0
    df = pd.DataFrame()
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            data = pd.read_csv(file_path, sep="\t", parse_dates=[2], infer_datetime_format=True, usecols=[0, 1, 2])
            print("preprocessing {}...".format(file_path))
            data = preprocess_data(data)
            save_df(data, fn)
            file_counter += 1
    return df


def alphanumeric_preprocessor(text):
    """
    Removes non alphanumeric characters and converts to lowercase
    :param text:
    :return:
    """
    try:
        return re.sub("\s\s+", " ", re.sub("[^a-z1-9]", " ", text.lower()))
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
