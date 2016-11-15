import pandas as pd
import os
import argparse
import re
from sklearn.feature_extraction.text import CountVectorizer

data_path = "../AOL-user-ct-collection/"

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
                [df, pd.read_csv(file_path, sep="\t", parse_dates=[2], infer_datetime_format=True, usecols=[0, 1, 2])])
            file_counter += 1
    return df


def alphanumeric_preprocessor(text):
    """
    Removes non alphanumeric characters and converts to lowercase
    :param text:
    :return:
    """
    try:
        return re.sub("[^a-z1-9]", " ", text.lower())
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


def users_from_data(df):
    return df.groupby('AnonID')


def sessions(df):
    session_list = []

    users = users_from_data(df)
    # for each user get all sessions and append to sessions list.
    for user in users:
        for s in get_sessions_from_user(user[1]):
            session_list.append(s)
            # print(session_list)


def session_generator(df):
    """
    :param df: dataframe to retrieve sessions from
    :return: sessions in the dataframe one by one.
    """
    users = users_from_data(df)
    for user in users:
        for s in get_sessions_from_user(user[1]):
            yield s


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
    user.loc[:, 'time_label'] = pd.Series(time_labels, index=user.index)
    user_sessions = user.groupby('time_label')
    for i, session in user_sessions:
        yield session


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nfiles", type=int, default=2,
                        help="number of files to process")
    args = parser.parse_args()

    data = read_files(args.nfiles)

    print "preprocessing..."
    data = preprocess_data(data)
    print(data.columns.values)
    print "get sessions..."
    sessions(data)
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)
    print "vectorizing..."
    print vectorizer.fit_transform(data["Query"].values)
    print data.head(10)


__main__()
