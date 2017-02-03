import os
os.chdir('D:/data-analysis/titanic')

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:, 'Pclass':'Embarked'], test.loc[:, 'Pclass':'Embarked']))

all_data.info()

###############################################################################
# 處理少量的missing data

# Embarked
all_data['Embarked'].value_counts()
all_data['Embarked'].fillna('S', inplace=True)

# Fare
all_data['Fare'].fillna(all_data['Fare'].mean(), inplace=True)

###############################################################################
# (Age) 處理大量的missing data

all_data['Title'] = all_data['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())

 # a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

# we map each title
all_data['Title'] = all_data['Title'].map(Title_Dictionary)
    
gropued = all_data.groupby(['Sex', 'Pclass', 'Title'])
gropued.median()

def fillAge(row):
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 30
        elif row['Title'] == 'Mrs':
            return 45
        elif row['Title'] == 'Officer':
            return 49
        elif row['Title'] == 'Royalty':
            return 39
    
    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return 20
        elif row['Title'] == 'Mrs':
            return 30
    
    elif row['Sex']=='female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return 18
        elif row['Title'] == 'Mrs':
            return 31
    
    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return 6
        elif row['Title'] == 'Mr':
            return 41.5
        elif row['Title'] == 'Officer':
            return 52
        elif row['Title'] == 'Royalty':
            return 40
    
    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return 2
        elif row['Title'] == 'Mr':
            return 30
        elif row['Title'] == 'Officer':
            return 41.5
    
    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return 6
        elif row['Title'] == 'Mr':
            return 26
    
all_data['Age'] = all_data.apply(lambda r : fillAge(r), axis=1) #axis=1, apply function to each row

all_data.info()

###############################################################################
# (Cabin) 處理大量的missing data
all_data['Cabin'].fillna('U', inplace=True)
all_data['Cabin'] = all_data['Cabin'].map(lambda c : c[0])

###############################################################################
# create new feature
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['Singleton'] = all_data['FamilySize'].map(lambda s : 1 if s==1 else 0)
all_data['SamllFamily'] = all_data['FamilySize'].map(lambda s : 1 if s>=2 and s<=4 else 0)
all_data['LagreFamily'] = all_data['FamilySize'].map(lambda s : 1 if s>4 else 0)

# all_data.shape[0] = 1309
a = all_data['Singleton'].value_counts()[1] + all_data['SamllFamily'].value_counts()[1] + all_data['LagreFamily'].value_counts()[1]
print a


###############################################################################
# drop columns
all_data.drop(['Name', 'Ticket'], inplace=True, axis=1)

###############################################################################
# one-hot encoding
pclass_dummies = pd.get_dummies(all_data['Pclass'], prefix='Pclass')
all_data = pd.concat([all_data, pclass_dummies], axis=1)
all_data.drop('Pclass', inplace=True, axis=1)

sex_dummies = pd.get_dummies(all_data['Sex'], prefix='Sex')
all_data = pd.concat([all_data, sex_dummies], axis=1)
all_data.drop('Sex', inplace=True, axis=1)

cabin_dummies = pd.get_dummies(all_data['Cabin'], prefix='Cabin')
all_data = pd.concat([all_data, cabin_dummies], axis=1)
all_data.drop('Cabin', inplace=True, axis=1)

embarked_dummies = pd.get_dummies(all_data['Embarked'], prefix='Embarked')
all_data = pd.concat([all_data, embarked_dummies], axis=1)
all_data.drop('Embarked', inplace=True, axis=1)

title_dummies = pd.get_dummies(all_data['Title'], prefix='Title')
all_data = pd.concat([all_data, title_dummies], axis=1)
all_data.drop('Title', inplace=True, axis=1)


###############################################################################
# feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(all_data)
scaler_all_data = scaler.transform(all_data)

x_train = scaler_all_data[:train.shape[0]]
x_test = scaler_all_data[train.shape[0]:]
y_train = train['Survived']

###############################################################################
# feature selection 用ExtraTreesClassifier看看自己create的參數有沒有貢獻
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(x_train, y_train)

features = pd.DataFrame()
features['feature'] = all_data.columns
features['importance'] = clf.feature_importances_
imp = features.sort(['importance'],ascending=False)
imp.plot(kind='barh')

#model = SelectFromModel(clf, prefit=True)
#x_train = model.transform(x_train)
#x_train.shape
#
#x_test = model.transform(x_test)
#x_test.shape

##############################################################################
# Model
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(np.sqrt(logs.get('loss')))
        
        if logs.get('val_loss') == None:
            #print 'rmse loss = %f' % (np.sqrt(logs.get('loss')))
            pass             
        else:
            self.val_losses.append(np.sqrt(logs.get('val_loss')))            
           
p = x_train.shape[1]
y_train = np_utils.to_categorical(y_train, 2)

def nn_model(): 
    
    drop = 0.05
    
    model = Sequential()

    model.add(Dense(1000, input_dim=p, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(900, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(800, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(700, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(600, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(500, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(400, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(300, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(200, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(100, activation='softplus'))
    model.add(Dropout(drop))
    
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

EPOCH = 200

nn = nn_model()
hist = nn.fit(x_train, y_train, nb_epoch=EPOCH, batch_size=10, validation_split=0.1, verbose=2)

plt.title('loss/val_loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.title('acc/val_acc')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.show()

# NN Model太大了 忽略不做...
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
#
#x_train = scaler_all_data[:train.shape[0]]
#y_train = train['Survived'].values
#kfold = StratifiedKFold(n_splits=10, shuffle=True)
#estimator = KerasClassifier(build_fn=nn_model, nb_epoch=EPOCH, batch_size=10, verbose=0)
#results = cross_val_score(estimator, x_train, y_train, cv=kfold)
#
## results: 0.818236
#print "results: %.6f" % (results.mean()) 
#
#y_train = np_utils.to_categorical(y_train, 2)
#nn.fit(x_train, y_train, nb_epoch=EPOCH, batch_size=10, verbose=2)

###############################################################################
# Output
output = nn.predict_classes(x_test)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output.to_csv('output.csv', index=False)
#Your submission scored 0.77033

###############################################################################
# lr model
from sklearn.linear_model import LogisticRegressionCV
x_train = scaler_all_data[:train.shape[0]]
y_train = train['Survived']
lr_cv = LogisticRegressionCV(cv=10)
lr_cv.fit(x_train, y_train)

# train score 0.8395
lr_cv.score(x_train, y_train)

# test score 0.7799
np.mean(lr_cv.scores_[1])

###############################################################################
# Output
output = lr_cv.predict(x_test)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output.to_csv('output.csv', index=False)
#Your submission scored 0.78469