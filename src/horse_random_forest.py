#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 00:46:32 2021

@author: kaiwaichan
"""

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

races_data = pd.read_csv(os.path.join(DATA_DIR, "races.csv"), index_col="race_id")
runs_data = pd.read_csv(os.path.join(DATA_DIR, "runs.csv"))

#Drop variables
df1 = races_data[['race_id', 'venue', 'config', 'distance', 'going', 'race_class']]
df1 = df1.fillna(0)

df2 = runs_data[['race_id', 'horse_no', 'horse_age', 'horse_country', 'horse_type', 'declared_weight', 'actual_weight', 'draw', 'win_odds', 'place_odds','won','result']]
df2 = df2.fillna(0)

#All matches
raceNum = max(races_data['race_id'])+1
win = np.empty((raceNum,14))
winVec = np.zeros(raceNum)
placeBinary = np.empty((raceNum,14))
result = np.empty((raceNum,14))
firstFour = np.empty((raceNum,14))
X = np.zeros((raceNum,14*8+5)) #14 horses x 8 variables + 5 match variables

#14 horses matches only
raceNum14 = df2['horse_no'].value_counts()[14]
X14 = np.zeros((raceNum14,14*8+5))
win14 = np.empty((raceNum14,14))
winVec14 = np.zeros(raceNum14)
placeBinary14 = np.empty((raceNum14,14))
result14 = np.empty((raceNum14,14))
firstFour14 = np.empty((raceNum14,14))

k = 0 #counter for adding data into X.14
is14 = False
#change venue to binary
df1.replace(['ST','HV'],[0,1],inplace=True)

#change config to numbers
config = df1['config'].unique().tolist()
df1.replace(config,[*range(1,len(config)+1)],inplace=True)

#change going into numbers
df1.replace(['FIRM','GOOD TO FIRM','GOOD','GOOD TO YIELDING','YIELDING','YIELDING TO SOFT','SOFT'],[*range(1,8,1)],inplace=True)
df1.replace(['FAST','WET FAST','WET SLOW','SLOW'], [1,1,7,7],inplace=True)

#Change horse country into numbers according to mean of result of each country
countrylist = df2['horse_country'].unique()
countrylist = np.delete(countrylist,np.where(countrylist==0))
averageRank = np.zeros(len(countrylist))
for i in range(len(countrylist)):
    countryRank = df2.loc[df2['horse_country'] == countrylist[i], 'result']
    averageRank[i] = np.mean(countryRank)

averageRank = np.argsort(averageRank) + 1
df2.replace(countrylist.tolist(), averageRank.tolist(),inplace=True)

#Change horse type into numbers according to mean of result of each type
typelist = df2['horse_type'].unique()
typelist = np.delete(typelist,np.where(typelist==0))
averageTypeRank = np.zeros(len(typelist))
for i in range(len(typelist)):
    typeRank = df2.loc[df2['horse_type'] == typelist[i], 'result']
    averageTypeRank[i] = np.mean(typeRank)

averageTypeRank = np.argsort(averageTypeRank) + 1
df2.replace(typelist.tolist(), averageTypeRank.tolist(),inplace=True)

for i in range(max(races_data['race_id'])+1):
#   race_i = df2.loc[df2['race_id']==i,].sort_values(by=['draw'])
    race_i = df2.loc[df2['race_id']==i,]
    horseNum = race_i.shape[0]


    #Add zero rows if not enough 14 horses
    if horseNum == 14:
        win14[k,] = race_i[['won']].T
        winVec14[k] = win14[k,].argmax() + 1
        result14[k,] = race_i[['result']].T
        for j in range(14):
            placeBinary14[k, j] = 1 if result14[k, j] <= 3 and result14[k, j] != 0 else 0
            firstFour14[k, j] = result14[k, j] if result14[k, j] <= 4 and result14[k, j] != 0 else 0
        is14 = True
    else: is14 = False

    while horseNum < 14:
        race_i = race_i.append(pd.Series(0, index=race_i.columns), ignore_index=True)
        horseNum = race_i.shape[0]

    #Extract different y
    win[i,] = race_i[['won']].T
    winVec[i] = win[i,].argmax()+1
    result[i,] = race_i[['result']].T
    for j in range(14):
        placeBinary[i,j] = 1 if result[i, j] <= 3 and result[i, j] != 0 else 0
        firstFour[i,j] = result[i,j] if result[i,j]<=4 and result[i,j]!=0 else 0

    #Further drop out columns
    race_i = race_i.drop(columns=['race_id','won','result','horse_no'])

    #Form vector x_i from df
    df = df1.loc[df1['race_id'] == i,].drop(columns='race_id')
    race = race_i.values.reshape(1,14*8)
    X[i,] = np.concatenate((df.values,race),axis=1)

    #Form vector x_i (14 horses)
    if is14 == True:
        X14[k,] = np.concatenate((df.values, race), axis=1)
        k += 1



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
### 1 : Directly Applying Random Forest to the dataset
X_train, X_test, y_train, y_test = train_test_split(X, winVec, test_size=0.3, random_state=4012)
regr = RandomForestRegressor(random_state=0)
regr.fit(X_train, y_train)
y_est = np.round(regr.predict(X_test))
ac1 = np.sum(y_est == y_test)/len(y_test) # Accuracy for guessing the first one in the race approx 0.084
print(ac1)

X_train, X_test, y3_train, y3_test = train_test_split(X, placeBinary, test_size=0.3, random_state=4012)
print(ac1)
ac2 = 0
for i in range(len(y_test)):
    ind = y_est[i].astype(int) - 1
    ac2 = ac2 + 1*(y3_test[i,ind].astype(int)==1)

ac2 = ac2/len(y_test) # Accuracy for guessing the first three in the race approx 0.2415
print(ac2)


# The accuracy against the number of bootstrapped tree
out = np.zeros(10)
for i in range(10):
    regr = RandomForestRegressor(n_estimators=(10*i+1),random_state=0)
    regr.fit(X_train, y_train)
    y_est = np.round(regr.predict(X_test))
    out[i] = np.sum(y_est == y_test) / len(y_test)
    # Result: array([0.1160105 , 0.09186352, 0.08976378, 0.07769029, 0.08031496,
    #        0.08766404, 0.09238845, 0.08346457, 0.08293963, 0.08503937])
print(out)


### 2 : Apply Random Forest to the Principal Components
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
PCs = pca.fit_transform(X_train.T)
X_tilde_train = pca.components_.T
regr.fit(X_tilde_train, y_train)
PCs = pca.fit_transform(X_test.T)
X_tilde_test = pca.components_.T
y_est = np.round(regr.predict(X_tilde_test))
ac3 = np.sum(y_est == y_test)/len(y_test) # Accuracy approx 0.082
print(ac3)
ac4 = 0
for i in range(len(y_test)):
    ind = y_est[i].astype(int) - 1
    ac4 = ac4 + 1*(y3_test[i,ind].astype(int)==1)
ac4 = ac4/len(y_test)
print(ac4)  # Accuracy approx 0.2236


# The accuracy against the number of PCs used
out2 = np.zeros(10)
for i in range(10):
    regr.fit(X_tilde_train[:,0:i+1], y_train)
    X_tilde_test = pca.components_.T[:,0:i+1]
    y_est = np.round(regr.predict(X_tilde_test))
    out2[i] = np.sum(y_est == y_test) / len(y_test)

print(out2) # Result: [0.07401575 0.06666667 0.0808399  0.09396325 0.08608924 0.09186352 0.07506562 0.08031496 0.07979003 0.08188976]