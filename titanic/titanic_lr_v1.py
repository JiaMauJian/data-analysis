# v1 - baseline model test

import os
os.chdir('D:/data-analysis/titanic')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:, 'Pclass':'Embarked'],
                      test.loc[:, 'Pclass':'Embarked']))

###############################################################################
#Cabin太多missing value
#Ticket資料沒什麼意義
all_data = all_data.drop(['Name', 'Cabin', 'Ticket'], axis=1)

###############################################################################
# Embarked = Port of Embarkation 你是從哪裡來的? 
# (C = Cherbourg; Q = Queenstown; S = Southampton)
# S最多
all_data['Embarked'].fillna('S', inplace=True)

###############################################################################
# Fare
all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)

###############################################################################
# Age
all_data['Age'].fillna(all_data['Age'].median(), inplace=True)


###############################################################################
# one-hot encoding
all_data = pd.get_dummies(all_data)

###############################################################################
# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(all_data)
all_data = scaler.transform(all_data)

###############################################################################
# model
# 先用最簡單的model，讓自己有個底 (baseline)後，再慢慢精進 (試feature, model...)
x_train = all_data[:train.shape[0]]
y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
lr.score(x_train, y_train)

###############################################################################
# cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

lr = LogisticRegression(random_state=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results = cross_val_score(lr, x_train, y_train, cv=kfold)

# 0.792408398223
print 'cv mean score = %f' % (np.mean(results))

###############################################################################
# run model with all data
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
lr.score(x_train, y_train)

###############################################################################
# submit
x_test = all_data[train.shape[0]:]
pred = lr.predict(x_test)
submission = pd.DataFrame(pred, index=test['PassengerId'], columns=['Survived'])
submission.to_csv('output.csv')

# Your submission scored 0.75598