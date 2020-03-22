import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# setting the working directory
os.chdir('/path/to/directory/')

# loading the csv file with each row corresponding to the .pgm fies in the dataset
table = pd.read_csv('all_mias_info.csv')

# recoding severity from char to int
table[' Severity'].loc[table[' Severity'] == 'B'] = 1
table[' Severity'].loc[table[' Severity'] == 'M'] = 2
table[' Severity'].loc[table[' Abnormality'] == 'NORM'] = 3

# creating dataframe for features (tabX) and array for label (taby)
tabX = table.loc[:,['ID', 'Character of tissue', ' Abnormality', ' X-Coordinate', ' Y-Coordinate', ' Radius']]
taby = table[' Severity']

# Creating training and test sets from the csv file
X_train0, X_test0, y_train0, y_test0 = train_test_split(tabX, taby, random_state=42, test_size=0.3)
