import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re as re

train = pd.read_csv('train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
    
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))

train = train.values
test  = test.values

from keras.callbacks import Callback
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
           
p = 7
x_train = train[0::, 1::]
y_train = np_utils.to_categorical(train[0::, 0], 2)

def create_model(): 
    
    drop = 0.3
    
    model = Sequential()

    model.add(Dense(400, input_dim=p, activation='softplus'))
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

model = create_model()
hist = model.fit(x_train, y_train, nb_epoch=350, batch_size=10, validation_split=0.1, verbose=2)

plt.title('loss/val_loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()

plt.title('acc/val_acc')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.show()

# =============================================================================   
# cross-validation
# 評估model的表現
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True)
estimator = KerasClassifier(build_fn=create_model, nb_epoch=350, batch_size=10, verbose=2)
x_train = train[0::, 1::]
y_train = train[0::, 0]
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print "mean score: %f" % (results.mean())
#mean score: 0.804789
#array([ 0.73333334,  0.82222222,  0.79775281,  0.7752809 ,  0.84269662,
#        0.87640449,  0.71910113,  0.80898876,  0.85393258,  0.81818182])


# =============================================================================   
# run model with all data
#model = create_model()
#model.fit(x_train, y_train, nb_epoch=350, batch_size=10, verbose=2)

preds = model.predict_classes(test)
test_data  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
pred_df = pd.DataFrame(preds, index=test_data["PassengerId"], columns=["Survived"])
pred_df.to_csv('output.csv', header=True, index_label='PassengerId')

# model.fit(x_train, y_train, nb_epoch=350, batch_size=10, verbose=2)
# 用全部的X再去train一次model，結果反而爛掉了???over-fitting???
# Your submission scored 0.76077

# model.fit(x_train, y_train, nb_epoch=350, batch_size=10, validation_split=0.1, verbose=2)	
#Your submission scored 0.81340

# 有做ModelCheckPoint，把val_loss最小的model存起來，再去預測後submit
# 結果反而沒比較好，可能原因是validation set 和 testing set的分佈本來就不一樣
#Your submission scored 0.80383