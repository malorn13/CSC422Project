import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("traffic-volume-counts-2012-2013.csv")
print(df.describe())
