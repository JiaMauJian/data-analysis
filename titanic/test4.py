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
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).sum())
    
for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 2, 'Q': 1} ).astype(int)
    
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


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train)
train = scaler.transform(train)

scaler = StandardScaler().fit(test)
test = scaler.transform(test)


from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import Nadam

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
    
    drop = 0.01
    
    model = Sequential()

    model.add(Dense(700, input_dim=p, activation='relu'))    
    model.add(Dropout(drop))    
    
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    
    return model

def create_model2(opt): 
    
    drop = 0.05
    
    model = Sequential()

    model.add(Dense(70, input_dim=p, activation='relu'))    
    model.add(Dropout(drop))    
    
    model.add(Dense(70, activation='relu'))    
    model.add(Dropout(drop))    
    
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model
    
model = create_model('adam')
hist_adam = model.fit(x_train, y_train, nb_epoch=1500, batch_size=10, verbose=2)
model.evaluate(x_train, y_train)

model = create_model()
hist_nadam = model.fit(x_train, y_train, nb_epoch=100, batch_size=10, verbose=2)
model.evaluate(x_train, y_train)

model = create_model(0.002)
hist_nadam2 = model.fit(x_train, y_train, nb_epoch=100, batch_size=10, verbose=2)
model.evaluate(x_train, y_train)

# [0.38939807972924073, 0.83052749672588944]

plt.title('loss')
plt.plot(hist_nadam.history['loss'], label='model1')
#plt.plot(hist_nadam2.history['loss'], label='model2')
plt.legend()
plt.show()

plt.title('acc')
plt.plot(hist_adam.history['acc'], label='adam')
plt.plot(hist_nadam.history['acc'], label='nadam')
plt.legend(loc=4)
plt.show()


model = create_model()
hist_nadam = model.fit(x_train, y_train, nb_epoch=10, batch_size=10, validation_split=0.1, verbose=2)
model.evaluate(x_train, y_train)

plt.title('nadam loss/val_loss')
plt.plot(hist_nadam.history['loss'], label='loss')
plt.plot(hist_nadam.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.title('nadam acc/val_acc')
plt.plot(hist_nadam.history['acc'], label='acc')
plt.plot(hist_nadam.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
# =============================================================================   
# cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True)
estimator = KerasClassifier(build_fn=create_model, nb_epoch=250, batch_size=10, verbose=2)
x_train = train[0::, 1::]
y_train = train[0::, 0]
results = cross_val_score(estimator, x_train, y_train, cv=kfold)
print "mean score: %f" % (results.mean()) #mean score: 0.810281

model = create_model()
model.fit(x_train, y_train, nb_epoch=10, batch_size=10, verbose=2)

preds = model.predict_classes(test)
test_data  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
pred_df = pd.DataFrame(preds, index=test_data["PassengerId"], columns=["Survived"])
pred_df.to_csv('output.csv', header=True, index_label='PassengerId')

	
# public score 0.81340
