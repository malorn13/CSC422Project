import time
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_filename = "./traffic-volume-counts-2012-2013.csv"

def main():

    #Simple function to open data with correct headers
    with open(dataset_filename, 'r') as f:
        headers = f.readline()[:-1].split(',')
        data = pd.read_csv(f, delimiter=',', names=headers)
        print(data)



if __name__ == '__main__':
    main()
