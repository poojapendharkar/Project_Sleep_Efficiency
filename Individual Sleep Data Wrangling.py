# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:16:33 2023

@author: penpo
"""


#Load the required packages

import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt


#load dataset
df = pd.read_csv('sleepdata_2.csv', delimiter=';')

#check if data is loaded properly
print(df.head())

#profile the dataset to perform exploratory data analysis
profile= ProfileReport(df, title="profile")
profile.to_file("sleepdata2EAD.html")
#based on the output of the above file we can remove mood and heart rate columns

print(df.isna().sum())
#we can see that airpressure, city, and mood columns have almost all or half of vales as na
# we drop those columns
#we keep column notes as is to create other columns from it later

df = df.drop(['Air Pressure (Pa)', 'City', 'Mood', 'Heart rate (bpm)'], axis=1)

print(df.dtypes)

# Subdividing the 'Sleep Notes' column into binary separate columns with True/False 
#we can see from EAD output file that notes has values like Ate late, Coffee, Tea, Worked out, Alcohol
df['Notes'].fillna('', inplace=True)

df['Caffeine'] = df['Notes'].apply(lambda x: True if ('tea' in x.lower()) or ('coffee' in x.lower()) else False)
df['Worked Out'] = df['Notes'].apply(lambda x: True if 'worked out' in x.lower() else False)
df['Ate late'] = df['Notes'].apply(lambda x: True if 'ate late' in x.lower() else False)
df['Alcohol'] = df['Notes'].apply(lambda x: True if 'alcohol' in x.lower() else False)


print(df.isnull().sum())

#Now we drop Notes column as we no longer need it
df = df.drop(['Notes'], axis=1)

#we still have na values in Window start and Window stop columns, lets drop those values
df = df.dropna()

#now we get all columns in correct values
#remove % from columns to keep it in numeric value
df['Regularity'] = df['Regularity'].str.replace('%', '')
df['Sleep Quality'] = df['Sleep Quality'].str.replace('%', '')

#compute and plot the correlation matrix
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#export clean dataset to csv
df.to_csv('individual_sleep_clean.csv', index=False)
