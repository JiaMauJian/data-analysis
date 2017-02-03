# v2 - l2 沒有什麼顯著的效果

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
# model & cross-validation
x_train = all_data[:train.shape[0]]
y_train = train['Survived']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

result_list = []
alphas = []

for alpha in np.arange(-5, 5):
    results = cross_val_score(LogisticRegression(C=10**alpha), x_train, y_train, cv=kfold)
    print 'alpha = %f, acc = %f' % (10**alpha, results.mean())
    alphas.append(10**alpha)
    result_list.append(results.mean())

#alpha = 0.100000, rmse = 0.798027

###############################################################################
# run model with all data   
lr = LogisticRegression(C=0.1)
lr.fit(x_train, y_train)
lr.score(x_train, y_train)

###############################################################################
# submit
x_test = all_data[train.shape[0]:]
pred = lr.predict(x_test)
submission = pd.DataFrame(pred, index=test['PassengerId'], columns=['Survived'])
submission.to_csv('output.csv')

# Your submission scored 0.75120