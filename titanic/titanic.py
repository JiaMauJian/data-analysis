import os
os.chdir('D:/data-analysis/titanic')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:, 'Pclass':'Embarked'],
                      test.loc[:, 'Pclass':'Embarked']))

all_data.info()

#Cabin太多missing value
#Ticket資料沒什麼意義
train = train.drop(['Cabin', 'Ticket'], axis=1)

# drop na (na都是Age，是否有更好的方式去fill na，如mean???)
train = train.dropna()

train.info()

# Embarked = Port of Embarkation 你是從哪裡來的? (C = Cherbourg; Q = Queenstown; S = Southampton)
# S最多
sns.countplot(x='Embarked', data=train)
train['Embarked'] = train['Embarked'].fillna('S')

embark_perc = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.countplot(x='Survived', hue='Embarked', data=train)
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S','C','Q'])






train.Survived.value_counts().plot(kind='bar')
plt.title("Distribution of Survival, (1 = Survived)")    
plt.show()

sns.barplot(x='Age', y='Survived', data=train)
plt.show()

plt.scatter(train.Survived, train.Age)
plt.show()

train.Pclass.value_counts().plot(kind='bar')
plt.title('Distribution of Class')

sns.distplot(train.Age[train.Pclass==1], label='Pclass=1')
sns.distplot(train.Age[train.Pclass==2], label='Pclass=2')
sns.distplot(train.Age[train.Pclass==3], label='Pclass=3')
plt.legend()

train.Embarked.value_counts().plot(kind='bar')
plt.title('Passengers per boarding location')






