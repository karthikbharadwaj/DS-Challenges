'''
__authors__ = "Ganesh Karthik"
'''

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def remove_zero_var(df):
    '''
    The function removes columns with zero variance columns
    '''
    sel = VarianceThreshold(0.8 * (1-0.8))
    sel.fit(df)



if __name__ == "__main__":

    test_file = 'test.csv'
    train_file = 'train.csv'
    #Reading input file into dataframes
    X_test = pd.DataFrame.from_csv(test_file)
    train_data = pd.DataFrame.from_csv(train_file)
    #Converting target to numeric variable
    Y_train = pd.Series([int(x.split('Class_')[1]) for x in train_data['target']])
    #Getting X_test and X_train
    X_train = train_data.drop('target',axis=1)

