# Python code explaining how to Binarize feature values 
#Importing Libraries 
import pandas as pd 
#Importing Data 
data_set = pd.read_csv('data.csv') 
print(data_set.head()) 
age = data_set.iloc[:, 1].values 
salary = data_set.iloc[:, 2].values 
print ("\nOriginal age data values : \n", age) 
print ("\nOriginal salary data values : \n", salary) 
# Binarizing values 
from sklearn.preprocessing import Binarizer 
 
x = age 
x = x.reshape(1, -1) 
y = salary 
y = y.reshape(1, -1) 
# For age, let threshold be 35, For salary, let threshold be 61000 
binarizer_1 = Binarizer(35) # Age below 35 is binarized to 0 
binarizer_2 = Binarizer(61000) # Salary below 61000 is binarized to 0 
 # Transformed features 
print ("\nBinarized age : \n", binarizer_1.fit_transform(x)) 
print ("\nBinarized salary : \n", binarizer_2.fit_transform(y))


import pandas as pd 
import numpy as np 
# Generate 20 random integers uniformly between 0 and 99 
small_counts = np.random.randint(0, 100, 20) 
print(small_counts) 
# Map to evenly spaced bins 0-9 by division 
print(np.floor_divide(small_counts, 10)) 
large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897, 44, 28, 7971, 
926, 12] 
# print(np.floor(np.log10(large_counts))) 
#Map the counts to quartiles into 4 bins (quartile) 
print(pd.qcut(large_counts, 4, labels=False)) 
#convert large_counts into series data 
large_counts_series = pd.Series(large_counts) 
# print(large_counts_series) 
print(large_counts_series.quantile([0.25, 0.5, 0.75])) 



import pandas as pd 
import numpy as np 
#Log Transform Example 
data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]}) 
data['log+1'] = (data['value']+1).transform(np.log) 
print(data) 
#Negative Values Handling. Note that the values are different 
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log) 
print(data) 



#Importing Libraries 
import pandas as pd 
# Sklearn library 
from sklearn import preprocessing 
#Import Data 
data_set = pd.read_csv('data.csv') 
print(data_set.head()) 
# here Features - Age and Salary columns 
# are taken using slicing 
# to handle values with varying magnitude 
x = data_set.iloc[:, 1:3].values 
print ("\nOriginal data values : \n", x) 
#MIN MAX SCALER 
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
# Scaled feature 
x_after_min_max_scaler = min_max_scaler.fit_transform(x) 
print ("\nAfter min max Scaling : \n", x_after_min_max_scaler) 
# STANDARDIZATION 
Standardisation = preprocessing.StandardScaler() 
# Scaled feature 
x_after_Standardisation = Standardisation.fit_transform(x) 
print ("\nAfter Standardisation : \n", x_after_Standardisation) 