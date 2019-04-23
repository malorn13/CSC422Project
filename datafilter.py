# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:01:57 2019

@author: Administrator
"""


import numpy as np
import pandas as pd

dataset_filename = "./traffic-volume-counts-2012-2013.csv"
filtered_filename = "filtered-traffic-volume-counts-2012-2013.csv"


def drop_cols(names, df):
    df.drop(names, axis=1, inplace=True)
    return df

def check_missing_data(df):
    return df.isnull().sum().sort_values(ascending=False)

def remove_outliers(df, means, headers):
    for i in range(1, len(df)):
        for j in range(1, df.shape[1]):
            if df.ix[i,j] > 6000:
                df.ix[i,j] = means[j]
    return df

def main():

    #Simple function to open data with correct headers
    with open(dataset_filename, 'r') as f:
        headers = f.readline()[:-1].split(',')
        data = pd.read_csv(f, delimiter=',', names=headers)
    
    #print(data.describe())

    unused_cols = ["Segment ID", "Roadway Name", "Date", "From", "To", "Direction"]
    data = drop_cols(unused_cols, data)
    
    headers = [e for e in headers if e not in unused_cols]
    
    print(check_missing_data(data))
    
    data.plot.box(grid='True')
    
    means = data.mean()
    
    data = remove_outliers(data, means, headers)
    
    data.to_csv(filtered_filename)
    
    data.plot.box(grid='True')

    print(data.describe())


if __name__ == '__main__':
    main()