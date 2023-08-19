# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:26:00 2023

@author: penpo
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv("individual_sleep_clean.csv")


#create a new column to check the amount of time spent in bed but not actually sleeping.

df['Time in bed non-sleep'] = df['Time in bed (seconds)'] - df['Time asleep (seconds)']

df.columns

#compute and plot the correlation matrix
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#now lets test for a few columns 
x = df[['Steps', 'Movements per hour', 'Time in bed non-sleep', 'Caffeine', 'Worked Out', 'Ate late', 'Alcohol']]
y = df['Sleep Quality']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=1)

model=LinearRegression()

model.fit(x_train, y_train)

y_pred= model.predict(x_test) 

rmse_test=mean_squared_error(y_test, y_pred, squared=False)

rmse_test
#rmse score is 11.2137

#############
#now lets test only for columns that were not derived from the notes
x = df[['Steps', 'Movements per hour', 'Time in bed non-sleep']]
y = df['Sleep Quality']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=1)

model=LinearRegression()

model.fit(x_train, y_train)

y_pred= model.predict(x_test) 

rmse_test=mean_squared_error(y_test, y_pred, squared=False)

rmse_test
#val is 11.213185 so the score hasn't improved much.


##################3
#we will now take all the columns and try feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
lr = LinearRegression()##model to be used for feature selection

df_fs = df.drop(['Start', 'End', 'Alarm mode', 'Window start', 'Window stop', 'Weather temperature (°C)', 'Weather type', 'Did snore'], axis=1)

df_fs['Caffeine'] = df_fs['Caffeine'].astype(int) # Transform boolean to integer    
df_fs['Worked Out'] = df_fs['Worked Out'].astype(int) # Transform boolean to integer    
df_fs['Ate late'] = df_fs['Ate late'].astype(int) # Transform boolean to integer  
df_fs['Alcohol'] = df_fs['Alcohol'].astype(int) # Transform boolean to integer           

x=df_fs.drop(columns=['Sleep Quality'])
y=df_fs['Sleep Quality']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
                                               random_state=1)
df_fs.shape[1]

sfs = SFS(lr, 
          k_features=(1,12), 
          forward=False, 
          scoring='neg_root_mean_squared_error',
          cv=5)

sfs.fit(x_train, y_train)##training here means finding important features based on RMSE

###what features were selected
sfs.k_feature_names_

#('Regularity', 'Time in bed (seconds)','Time asleep (seconds)','Time before sleep (seconds)','Snore time','Alcohol', 'Time in bed non-sleep')

##transformed data will have only selected features
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
lr.fit(X_train_sfs, y_train)
y_pred = lr.predict(X_test_sfs)
'''rmse with backward/forward method'''

rmse = mean_squared_error(y_test, y_pred, squared=False)
print ("rmse is",rmse)

#val is 8.96768109324963
#with backward elimination and feature selection the score has improved to 8.96

# Get the coefficients and intercept
coefficients = lr.coef_
intercept = lr.intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

#we get y = 9.7521 + 0.0657 * Regularity + 10189767400 * Time in Bed - 10189767400 * Time  asleep  
# 0.0023 * Time before sleep - 0.0028 * Snore time - 10.9925 * Alcohol - 10189767400 * Time in bed non-sleep

#if we predict the sleep quality for a person who sleeps regularly (say 90% of times), time in bed and time asleep is 27774.0, 26755.6
#Time before sleep is 500, a high snorer (500), Does not drink alcohol and spends just 1 min non sleep time in bed

lr.predict([[90, 27774.0, 26755.6, 500, 500, 0, 60 ]])
#val is  array([9.76587305e+12]) so sleep quality is about 97%

#if the same person spends 6mins in bed without sleeping so maybe on phone or stressed
lr.predict([[90, 27774.0, 26755.6, 500, 500, 0, 360 ]])
#we can see that val is array([6.70894284e+12]) the sleep quality has reduced to about 67%

#################
#Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df_fs)
scaled_df=scaler.transform(df_fs)

wcv=[]
silk_score=[]

for i in range (2,14):
    km= KMeans(n_clusters=i, random_state= 0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))


#plot wcv
plt.plot(range(2,14), wcv)
plt.xlabel('No of clusters')
plt.ylabel('Wihtin cluster vairation')

#plot silk score
plt.plot(range(2,14), silk_score)
plt.xlabel('No of clusters')
plt.ylabel('Silk Score')


#we get peak at 3 so we go with 3 clusters
km3= KMeans(n_clusters=3, random_state=0)
km3.fit(scaled_df)

df_fs['labels']=km3.labels_

#interpret cluster
df_fs.groupby('labels').mean() 

#display settings
pd.options.display.max_rows = None
pd.options.display.max_columns = None

#cluster 0
c0 = df_fs.loc[df_fs['labels']==0]
c0.describe()
#Sleep quality is highest if Regularity is high, movements per hour are low 

#cluster 1
c1 = df_fs.loc[df_fs['labels']==1]
c1.describe()
# this is most favorable cluster as it shows that Sleep quality is high if person has worked out and had maximum steps that day
#although it is also showing relation to Ate late and caffenie we can assume that it might be because the person recorded those things in note that day 

#cluster 2
c2 = df_fs.loc[df_fs['labels']==2]
c2.describe()