import os
import pandas as pd

if os.path.isfile('lamdamart_data_next_query.csv'):
    print "read csv!"
    df = pd.read_csv('lamdamart_data_next_query.csv')
    print(df)
    df.drop('Unnamed: 0', 1)
    lambdamart_data = df.get_values()[:,1:]
    print lambdamart_data
    print "loaded!!!!"
    # lambdaMart(lambdamart_data, experiment_string)