import pandas as pd
import os

data_path = "../AOL-user-ct-collection/"

'''
    Sessions end after 30 minutes of inactivity, same definition google uses
    https://support.google.com/analytics/answer/2731565?hl=en
'''
session_inactivity = 30


def read_files(max_files=10) -> pd.DataFrame:
    file_counter = 0
    df = pd.DataFrame()
    for fn in os.listdir(data_path):
        file_path = data_path + fn
        if os.path.isfile(file_path) and fn.startswith('user-ct-test-collection') and file_counter < max_files:
            print("loading {}...".format(file_path))
            df = pd.concat([df, pd.read_csv(file_path, sep="\t", parse_dates=[2], infer_datetime_format=True)])
            file_counter += 1
    return df


def users_from_data(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return df.groupby('AnonID')


def sessions(df: pd.DataFrame):
    session_list = []

    users = users_from_data(df)
    # for each user get all sessions and append to sessions list.
    for user in users:
        print(user)
        session_list.append(get_session_from_user(user))


def get_session_from_user(user):
    session = pd.DataFrame
    for row in user[1].iterrows():
        print(type(row[1]['QueryTime']))
        print(row[1]['QueryTime'])
    return session


data = read_files(1)
print(data.columns.values)
sessions(data)
