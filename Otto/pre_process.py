'''
__authors__ = "Ganesh Karthik"
'''

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split


def remove_zero_var(df):
    '''
    The function removes columns with zero variance columns
    Removes data with
    '''

    sel = VarianceThreshold(0.8 * (1-0.8))
    return pd.DataFrame(sel.fit_transform(df))



if __name__ == "__main__":

    test_file = 'test.csv'
    train_file = 'train.csv'
    #Reading input file into dataframes
    X_test = pd.DataFrame.from_csv(test_file)
    train_data = pd.DataFrame.from_csv(train_file)
    #Converting target to numeric variable
    Y_data = pd.Series([int(x.split('Class_')[1]) for x in train_data['target']])
    #Removing target to get X_train
    X_data = train_data.drop('target',axis=1)
    X_data = remove_zero_var(X_data)
    #splitting for cross validation
    x_train, x_test, y_train, y_test = train_test_split(X_data,Y_data,test_size=0.4)
    print y_train



