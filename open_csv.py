import time
import csv

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

dataset_filename = "./traffic-volume-counts-2012-2013.csv"

def main():

    #Simple function to open data with correct headers
    with open(dataset_filename, 'r') as f:
        headers = f.readline()[:-1].split(',')
        data = pd.read_csv(f, delimiter=',', names=headers)

    x = pd.DataFrame(data['ID'])
    y = pd.DataFrame(data['5:00-6:00PM'])

    model = LinearRegression()
    kfold = KFold(n_splits=10)
    
    scores = []
    for i, (train,test) in enumerate(kfold.split(x, y)):
        model.fit(x.iloc[train,:], y.iloc[train,:])
        score = model.score(x.iloc[test,:], y.iloc[test,:])
        scores.append(score)
    print(scores)
    #Testing different sklearn methods



if __name__ == '__main__':
    main()
