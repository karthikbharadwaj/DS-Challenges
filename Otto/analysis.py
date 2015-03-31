'''
__authors__ = "Ganesh Karthik"
'''

import pandas as pd
from sklearn.cross_validation import train_test_split
from scripts import pre_process as pre_pros,feature_selection as FS









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
    X_data = pre_pros.remove_zero_var(X_data)
    X_reduced = pre_pros.dimension_reduction(X_data,dim=2)
    #splitting for cross validation
    x_train, x_test, y_train, y_test = train_test_split(X_data,Y_data,test_size=0.4)



