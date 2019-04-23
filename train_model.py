import time
import csv

import numpy as np
import pandas as pd

#import sklearn as sklearn
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset_filename =     "F:/CSC Development/CSC422/ProjectGit/CSC422Project/traffic-volume-counts-2012-2013.csv"
new_dataset_filename = "F:/CSC Development/CSC422/ProjectGit/CSC422Project/expanded-traffic-counts.csv"

def main():

    #Simple function to open data with correct headers
    with open(dataset_filename, 'r') as f:
        full_headers = f.readline()[:-1].split(',')
        full_data = pd.read_csv(f, delimiter=',', names=full_headers)

    with open(new_dataset_filename, 'r') as f:
        headers = f.readline()[:-1].split(',')
        data = pd.read_csv(f, delimiter=',', names=headers)
    
    data = remove_outliers(data)
    data = shuffle(data)
    
    X = data.iloc[:,0:2].values
    y = data.iloc[:,2].values

    print(X[X == 1])

    rng = np.random.RandomState(55)
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=14), n_estimators=300, random_state=rng)
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    #pipe_tree = make_pipeline(DecisionTreeRegressor(random_state=1))
    #gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, cv=10)

    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X[train,:], y[train])
        score = model.score(X[test,:], y[test])
        scores.append(score)
    print(scores)
    
    #model.predict(X.iloc[0].values.reshape(-1,1))
    
    y_1 = model.predict(X)
    
    plt.figure()
    train_scores, valid_scores = validation_curve(DecisionTreeRegressor(), X, y, "max_depth", np.arange(5,25))
    #print(train_scores)
    print(valid_scores)
    #plt.plot(np.arange(5,25), train_scores, label="Training Scores")
    plt.plot(np.arange(5,25), valid_scores, label='Folds')
 
    plt.xlabel("Max Tree Depth")
    plt.ylabel("Score")
    plt.title("Max Depth of Decision Tree Analysis")
    plt.legend()
    plt.show()

    
    

def drop_cols(names, df):
    df.drop(names, axis=1, inplace=True)
    return df

def check_missing_data(df):
    return df.isnull().sum().sort_values(ascending=False)

def remove_outliers(df):
    for i in range(1, len(df)):
        for j in range(1, df.shape[1]):
            if df.iloc[i,j] > 6000:
                df.iloc[i,j] = df.iloc[i,j-1]
    return df
if __name__ == '__main__':
    main()
