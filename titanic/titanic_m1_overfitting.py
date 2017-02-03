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
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
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
hist = model.fit(x_train, y_train, nb_epoch=150, batch_size=10, validation_split=0.1, verbose=2)

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

model = KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
y_train_no_categorical = train['Survived'].values
results = cross_val_score(model, x_train, y_train_no_categorical, cv=kfold)

print 'acc = %f' % (results.mean()) #0.820385648828

# =============================================================================
# 確定cross-validation驗證完model後，把全部的X再拿進去train一次
model = create_model()
model.fit(x_train, y_train, nb_epoch=150, batch_size=10, verbose=0)

x_test = all_data[train.shape[0]:]
preds = model.predict_classes(x_test)
pred_df = pd.DataFrame(preds, index=test["PassengerId"], columns=["Survived"])
pred_df.to_csv('output.csv', header=True, index_label='PassengerId')

# public scrore 0.73206