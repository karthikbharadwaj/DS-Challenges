__author__ = 'karthikb'

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import re


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

def process_data(file):
    '''
    The function pre processes the training file
    :param file: Training_Data.csv
    :return None
    '''
    #Regular expression to detect opinion values eg. Very good(6)
    pattern = re.compile('.*\(\d+\).*')
    with open(file,'r') as input:
        count = 0
        header = input.next().split(',')
        for line in input:
            words = line.strip().split(',')
            counter = 0
            for word in words:
                word = word.strip("\"")
                original_word = word
                value = pattern.match(word)
                if value:
                    word = value.group()
                    #Extracting number from matched pattern
                    value = int(re.search('\d',word).group())
                    #Setting the number
                    words[counter] = value
                if is_number(word):
                    words[counter] = float(word)
                counter += 1
            count += 1
            print words
            #Testing for first 20 lines
            if count == 20:
                break
process_data('Training_Data.csv')


