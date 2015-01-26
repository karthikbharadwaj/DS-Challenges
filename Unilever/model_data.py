__author__ = 'karthikb'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processing import factorize_non_numeric_data,normalize
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error


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
    X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
    #Normalizing the data
    X_train,y_train = normalize(X_train,y_train)
    X_test,y_test = normalize(X_test,y_test)
    model = ExtraTreesClassifier(compute_importances=True)
    model.fit(X_train,y_train)
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    print "Feature importance:"
    i=1
    for f,w in zip(df.columns[sorted_idx], feature_importance[sorted_idx]):
        print "%d) %s : %d" % (i, f, w)
        i+=1
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    nb_to_display = 30
    plt.barh(pos[:nb_to_display], feature_importance[sorted_idx][:nb_to_display], align='center')
    plt.yticks(pos[:nb_to_display], df.columns[sorted_idx][:nb_to_display])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    #logit = LogisticRegression(penalty='l2',C=1.0)
    #logit.fit(X_train,y_train)
    #y_pred = logit.predict(X_test)
    #print mean_squared_error(y_test,y_pred)

if __name__ == '__main__':
    model_data('processed_train.csv')
