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
# Alone or Family (獨身一人是否比較容易存活?)
family = train[['Survived', 'Parch', 'SibSp']]
tmp = train['Parch'] + train['SibSp']
family.loc[:, 'IsFamily'] = ( tmp > 0 ) * 1
sns.barplot(x='IsFamily', y='Survived', data=family, order=[1,0]) # 結果有家庭的存活率高
# 或許可以加新變數HasFamilyYN
all_data.loc[family['IsFamily'] == 1, 'HasFamilyYN'] = 1
all_data['HasFamilyYN'].fillna(0, inplace=True)

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
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

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
 
            
p = all_data.shape[1]
x_train = all_data[:train.shape[0]]
y_train = np_utils.to_categorical(train['Survived'], 2)

def create_model(): 
    
    model = Sequential()

    model.add(Dense(100, input_dim=p, activation='relu', name='1st hiddne layer'))

    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_model()
hist = model.fit(x_train, y_train, nb_epoch=30, batch_size=10, validation_split=0.1, verbose=2)

plt.title('loss/val_loss [over-fitting]')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.title('acc/val_acc')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.show()

###############################################################################
# cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

model = KerasClassifier(build_fn=create_model, nb_epoch=30, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
y_train_no_categorical = train['Survived'].values
results = cross_val_score(model, x_train, y_train_no_categorical, cv=kfold)

print 'acc = %f' % (results.mean()) #0.817053

# =============================================================================
# 確定cross-validation驗證完model後，把全部的X再拿進去train一次
# acc跳動太大，用EarlyStopping很難控制，乾脆用ModelCheckpoint，把最好的結果存起來
model = create_model()
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(x_train, y_train, nb_epoch=300, batch_size=10, validation_split=0.1, callbacks=callbacks_list, verbose=2)
scores = model.evaluate(x_train, y_train, verbose=0)
print scores

x_test = all_data[train.shape[0]:]
preds = model.predict_classes(x_test)
pred_df = pd.DataFrame(preds, index=test["PassengerId"], columns=["Survived"])
pred_df.to_csv('output.csv', header=True, index_label='PassengerId')

# public scrore  0.74163