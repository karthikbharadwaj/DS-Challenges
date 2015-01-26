__author__ = 'karthikb'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processing import factorize_non_numeric_data,normalize
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from feature_selection import feature_selection_plot
from sklearn.metrics import mean_squared_error,accuracy_score
import sklearn.cross_validation as cross_validation



def model_data(process_file):
    df = pd.read_csv('processed_train.csv')
    #Removing unwanted columns
    personal_particulars = df.columns[266:277]
    df = df.drop(personal_particulars,axis=1)
    #Factorizing the columns
    non_numeric_columns =  df.loc[:,df.dtypes == np.object].columns
    df[non_numeric_columns] = factorize_non_numeric_data(df,non_numeric_columns)
    df = df.fillna(0)
    #Target variable
    y =  df['Overall.Opinion']
    x = df.drop('Overall.Opinion',axis=1)
    #Splitting the data into train and test
    x = feature_selection_plot(x,y,no_of_features=40)
    X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
    #Normalizing the data
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    logit = LogisticRegression(penalty='l2',C=1.0)
    svm = LinearSVC(penalty='l2',C=1.0)
    logit.fit(X_train,y_train)
    values = cross_validation.cross_val_score(logit, X_test, y_test, cv=5,score_func=mean_squared_error)
    print values.mean()
    values = cross_validation.cross_val_score(svm, X_test, y_test, cv=5,score_func=mean_squared_error)
    print values.mean()
    #print '%f' % mean_squared_error(y_test,y_pred)

if __name__ == '__main__':
    model_data('processed_train.csv')
