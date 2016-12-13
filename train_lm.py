import lambda_mart as lm
import pandas as pd
import os
from datetime import datetime

experiment = 'next_query'
file_name = '../data/lamdamart_data_next_query.csv'
file_name_val = '../data/lamdamart_data_next_query_val.csv'
file_name_test = '../data/lamdamart_data_next_query_test.csv'

time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Loading sessions feature file...]" % time)
if os.path.isfile(file_name):
    df = pd.read_csv(file_name)
    df_val = pd.read_csv(file_name_val)
    df_test = pd.read_csv(file_name_test)
else:
    print('Could not find ' + file_name)
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Loaded train features...]" % time)

df = df.drop('Unnamed: 0', 1)
df_val = df_val.drop('Unnamed: 0', 1)
df_test = df_test.drop('Unnamed: 0', 1)
lambdamart_data = df.get_values()
lambdamart_data_val = df_val.get_values()
lambdamart_data_test = df_test.get_values()
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Training LambdaMART with HRED features]" % time)
lm.lambdaMart(lambdamart_data, lambdamart_data_val, lambdamart_data_test, experiment + '_HRED')

df = df.drop('HRED', 1)
df_val = df_val.drop('HRED', 1)
df_test = df_test.drop('HRED', 1)
lambdamart_data = df.get_values()
lambdamart_data_val = df_val.get_values()
lambdamart_data_test = df_test.get_values()
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Training LambdaMART without HRED features]" % time)
lm.lambdaMart(lambdamart_data, lambdamart_data_val, lambdamart_data_test, experiment)