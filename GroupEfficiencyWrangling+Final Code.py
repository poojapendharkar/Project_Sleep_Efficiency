#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 21:57:23 2023

"""

import pandas as pd 
import seaborn as sns

df=pd.read_csv('Sleep_Efficiency.csv')

df2=df.drop_duplicates(subset=None)

mean_value=df['Caffeine consumption'].mean()
mean_value= round(mean_value)
df["Caffeine consumption"].fillna(value=mean_value, inplace=True)

alcmean_value=df['Alcohol consumption'].mean()
alcmean_value= round(alcmean_value)
df["Alcohol consumption"].fillna(value=alcmean_value, inplace=True)

awakemean_value=df['Awakenings'].mean()
awakemean_value= round(awakemean_value)
df["Awakenings"].fillna(value=awakemean_value, inplace=True)

exercisemean_value=df['Exercise frequency'].mean()
exercisemean_value= round(exercisemean_value)
df["Exercise frequency"].fillna(value=exercisemean_value, inplace=True)


df3= df.dropna() #no more missinng values so we can still use df in our analysis

df.to_csv("sleep_efficiency_clean.csv")


heatmap= sns.heatmap(correlation, annot=True)




#drop the following columns because they aren't helpful to our analysis...these columns were helpful in determining sleep duration 
df2= df.drop(['Bedtime', 'Wakeup time'], axis=1)

#get dummy variables for Smoking Status and Gender
df3= pd.get_dummies(df2, drop_first=True)

#run corrrelation
correlation= df.corr()

#drop highly correlated variables
df4= df3.drop(['ID','Deep sleep percentage', 'Light sleep percentage'], axis=1)

#recheck corrrrelation 
correlation= df4.corr()

#REGRESSION MODEL
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#Exploratative Linear Regression 
x=df4.drop('Sleep efficiency', axis=1)
y=df3['Sleep efficiency']

model.fit(x, y)

pd.DataFrame(zip(x.columns, model.coef_))

import statsmodels.api as sm
from scipy import stats

X2=sm.add_constant(x)
est= sm.OLS(y, X2)
est2=est.fit()
print(est2.summary()) #R-squared= 0.5

#Awakenings Signficant at 0.001
#Alcohol consumotion at 0.001
#Smoking Status at 0.001
#Excercise frequency at 0.001
#age at 0.05

#If awakenings increase by 1, then sleep efficiency decreases by 0.0488
#If alcohol consumption increases by 1 unit (1 oz) sleep effiiciency decreases by 0.0229
#If age incresaes by 1 year, sleep efficiency increases by 0.0009
#If excersise frequency increases by 1 unit (# of timies excersisied/week), sleep efficiency increases b7 0.0120
#Smokers sleep efficiency is 0.0824 less than non-smokers


#making a prediction model using these variables 
x=df4.drop('Sleep efficiency', axis=1)
y=df4['Sleep efficiency']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression 
lm=LinearRegression() 
lm.fit(x_train, y_train)
predictions=lm.predict(x_test)
from sklearn.metrics  import mean_squared_error
mse=mean_squared_error(y_test, predictions)
rmse=mse**0.5
print(rmse)
#rmse= 0.099

samplepred= lm.predict([[65,6,18, 0,0,0,3,0,1]])
print(samplepred) #0.86


#CLUSTERING


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


df=pd.read_csv('updated_sleep_efficiency.csv')
df2= df.drop(['Unnamed: 0', 'Gender', 'ID','Smoking status', 'Bedtime', 'Wakeup time'], axis=1)

#data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df2)
scaled_df=scaler.transform(df2)


km4=KMeans(n_clusters=4, random_state=0) #initialize
km4.fit(scaled_df) #train

#wcv and silk score
km4.inertia_ #wcv (withing cluster variation) 
silhouette_score(scaled_df, km4.labels_) ##silk score

wcv=[]
silk_score=[]

for i in range (2, 10): 
    km=KMeans(n_clusters=i, random_state=0)
    km.fit(scaled_df)
    #hcv and silk score
    wcv.append(km.inertia_) #hcw
    silk_score.append(silhouette_score(scaled_df, km.labels_))


plt.plot(range(2,10),wcv)
plt.xlabel("No of clusters")
plt.ylabel('Within Cluster Variaation ')


plt.plot(range(2,10),silk_score)
plt.xlabel("No of clusters")
plt.ylabel('Silk Score')

km3=KMeans(n_clusters=3, random_state=0)
km3.fit(scaled_df)

df['labels']= km3.labels_

df3= df.groupby('labels').mean()


#Cluster 1 (Label 0)
#Middle range sleep efficiency w/most awakenings, lowest caaffeine consumption, and some excersise weekly 
#Cluster 2 (Label 1)
#Highest sleep efficiency, w same deep sleep percentage as 1 but much less aawakenings...more caaffeinie consumption and less alcohol consumption...more excersiise
#Cluster 3 (Label 2) 
#Worst sleep efficiency w/much less deep sleep percentage and more light sleep, more awakenings than 2, most alcohol consumption, least excersise



#*** checking to see if model improves without mean imputation
import pandas as pd
df=pd.read_csv('Sleep_Efficiency.csv')

df2=df.drop_duplicates(subset=None)
df3= df.dropna()


df4= df3.drop(['Bedtime', 'Wakeup time'], axis=1)

df4= pd.get_dummies(df3, drop_first=True)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

#seperate data into x and y 
x=df4[['Age', 'Gender_Male', 'Caffeine consumption', 'Alcohol consumption', 'Smoking status_Yes', 'Exercise frequency']]
y=df4['Sleep efficiency']

model.fit(x, y)

pd.DataFrame(zip(x.columns, model.coef_))

import statsmodels.api as sm
from scipy import stats

X2=sm.add_constant(x)
est= sm.OLS(y, X2)
est2=est.fit()
print(est2.summary())





