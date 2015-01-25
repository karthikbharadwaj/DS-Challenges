__author__ = 'karthikb'

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sps
import os,csv
import itertools
import re
from sklearn.metrics import mean_squared_error


def is_number(s):
    '''
    The function returns True if the string is numeric/float
    else False
    :param s:
    :return: boolean
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_data(input_file,output_file):
    '''
    The function pre processes the training file
    :param file: Training_Data.csv
    :return None
    '''
    #Regular expression to detect opinion values eg. Very good(6)
    pattern = re.compile('.*\(\d+\).*')
    #Removing output file if exists
    input = csv.reader(open(input_file,'rb'))
    output = csv.writer(open(output_file,'wb'))
    #reading the header line
    header = input.next()
    output.writerow(header)
    for words in input:
        word_counter = 0
        for word in words:
            word = word.strip("\"")
            original_word = word
            value = pattern.match(word)
            #Value of Ingredients from 2:154
            if 2 <= word_counter < 155:
                if word == 'NA':
                    words[word_counter] = 0
                else:
                    words[word_counter] = float(word)
            #Encoding Problem variables
            if 158 <= word_counter <= 228 or 277 <= word_counter <= 297 or 299 <= word_counter <= 300:
                try:
                    if word == 'NA':
                        words[word_counter] = 0
                    elif pattern.match(word):
                        value = int(re.search('\d',word).group())
                        words[word_counter] = value
                    else:
                        words[word_counter] = float(word)
                except ValueError:
                    words[word_counter] = word
            if 229 <= word_counter <=254:
                #print words[word_counter],header[word_counter]
                if word == 'No' or word == 'NA':
                    words[word_counter] = 0
                else:
                    words[word_counter] = 1
            word_counter += 1
        #writing the output file
        output.writerow(words)
    return


def factorize_non_numeric_data(df,columns):
    '''
    FActorizing the non_numeric_columns
    '''
    old_dataframe = df[columns]
    #converting to key,value pairs
    dict_frame =  old_dataframe.to_dict('records')
    new_frame = old_dataframe.apply(lambda x: pd.factorize(x)[0])
    return new_frame

def normalize(x,y):
    '''
    Normalize the numpy arrays
    :param x:
    :return normalized array:
    '''
    norm1 = x / np.linalg.norm(x)
    norm2 = y / np.linalg.norm(y)
    return norm1,norm2

def dimension_reduction(x,dim=None):
    pca = PCA(n_components=None)
    pca.fit_transform(x)
    reduced_data = pca.transform(x)
    return reduced_data


if __name__ == "__main__":
    process_data('Training_Data.csv','processed_train.csv')








