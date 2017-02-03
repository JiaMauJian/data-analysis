import os
os.chdir('D:/data-analysis/titanic')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re as re

train = pd.read_csv('train.csv')

###############################################################################
# Age 死活分布
age = train.loc[:, ['Age', 'Survived']]
age.dropna(inplace=True)
Survived = age.loc[age['Survived']==1, 'Age']
Dead = age.loc[age['Survived']==0, 'Age']
plt.hist([Survived, Dead], stacked=True, color = ['g','r'], bins = 10,label = ['Survived','Dead'])

###############################################################################
# Women and children first ! (小孩和女生存活率高)
person = train.loc[:, ['Age', 'Sex', 'Survived']]
person.loc[person['Age'] < 16, 'Person'] = 'child'
person.loc[(person['Age'] >= 16) & (person['Sex'] == 'male'), 'Person'] = 'male'
person.loc[(person['Age'] >= 16) & (person['Sex'] == 'female'), 'Person'] = 'female'
sns.barplot('Person', 'Survived', data=person)

###############################################################################
# Passengers with cheaper ticket fares are more likely to die (票價高=有錢人有地位=比較容易獲救?)
fare = train.loc[:, ['Fare', 'Survived']]
fare_die = fare.loc[fare['Survived']==0, 'Fare']
fare_sur = fare.loc[fare['Survived']==1, 'Fare']
plt.hist([fare_die, fare_sur], stacked=True, bins=30, label=['died', 'survived'])
plt.legend()
plt.show()

sns.boxplot(x='Survived', y='Fare', data=fare)
plt.ylim(0, 200)

###############################################################################
# 所以頭等艙票價高阿
pclass = train.loc[:, ['Pclass', 'Fare']]
sns.barplot(pclass['Pclass'], pclass['Fare'])

###############################################################################
# embarkation site (從哪邊登船跟存活有關係嗎??)
embark = train.loc[:, ['Embarked', 'Survived']]
sns.barplot('Embarked', 'Survived', data=embark)

###############################################################################
# Family Size (有家族的存活率高)
family = train.loc[:, ['Parch','SibSp','Survived']]
family.loc[ (family['Parch']+family['SibSp'] > 0), 'IsFamily'] = 1
family.loc[ (family['Parch']+family['SibSp'] == 0), 'IsFamily'] = 0
sns.barplot('IsFamily', 'Survived', data=family)

###############################################################################
# Title 跟 存活率
name = train.loc[:, ['Name', 'Survived']]

# a = name.loc[0, 'Name']
# a.split(',')[1].split('.')[0].strip()
# Out[79]: 'Mr'

name.loc[:, 'Title'] = name['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())
#name.loc[:, 'Title'] = name['Name'].map(lambda name:re.search(' ([A-Za-z]+)\.', name).group(1))

sns.barplot('Title', 'Survived', data=name, size = 2)

