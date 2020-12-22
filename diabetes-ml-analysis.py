"""
This is a simple ML analysis on a diagonistic diabetes data set.
thanks to user @visabh123 for providing the framework to allow for this analysis.
author: https://wwww.github.com/kevingabayan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('diabetes.csv')
# X = features
X = dataset.iloc[:, 0:8].values
# Y = what we are going to predict
Y = dataset.iloc[:, 8].values

dataset.groupby('Outcome').hist(figsize=(12, 12))

print("WIP")



