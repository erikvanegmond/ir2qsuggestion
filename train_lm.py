import lambda_mart as lm
import pandas as pd
import os
from datetime import datetime

experiment = 'next_query'
file_name = '../data/lamdamart_data_' + experiment + '.csv'

time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Loading sessions feature file...]" % time)
if os.path.isfile(file_name):
    df = pd.read_csv(file_name)
else:
    print('Could not find ' + file_name)
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Loaded train features...]" % time)

df = df.drop('Unnamed: 0', 1)
lambdamart_data = df.get_values()
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Training LambdaMART with HRED features]" % time)
lm.lambdaMart(lambdamart_data, experiment + '_HRED')

df = df.drop('HRED', 1)
lambdamart_data = df.get_values()
time = datetime.now().strftime('%d-%m %H:%M:%S')
print("[%s: Training LambdaMART without HRED features]" % time)
lm.lambdaMart(lambdamart_data, experiment)