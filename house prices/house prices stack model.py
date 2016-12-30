import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

import os
print os.getcwd()

train  = pd.read_csv("./house prices/train.csv")
test = pd.read_csv("./house prices/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# =============================================================================
# Data Preprocessing
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.median())
    
train["SalePrice"] = np.log1p(train["SalePrice"])

scaler = StandardScaler().fit(all_data[numeric_feats])
all_data[numeric_feats] = scaler.transform(all_data[numeric_feats])
    
#creating matrices for sklearn:
X = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# =============================================================================
# Lasso
#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import Lasso, LassoCV
model_lasso = Lasso()

from sklearn.model_selection import cross_val_score

rmse = []

def rmse_cv(X, y):
    rmse_list = []
    alphas = []
    for alpha in np.arange(-6, 1):    
        rmse = np.sqrt(-cross_val_score(Lasso(alpha=10**alpha), X, y, scoring="neg_mean_squared_error", cv = 10))
        print 'alpha = %f, rmse = %f' % (10**alpha, rmse.mean())    
        alphas.append(10**alpha)
        rmse_list.append(rmse.mean())
    return rmse_list, alphas

rmse, alphas = rmse_cv(X, y)

cv_lasso = pd.Series(rmse, index=alphas)
cv_lasso.plot(logx=True)
# bset alpha = 0.001000, rmse = 0.122480

# =============================================================================
# 確定cross-validation驗證完model後，把全部的X再拿進去train一次
model_lasso = Lasso(0.001)
model_lasso.fit(X, y)
#lasso model
#cv_10, alpha = 0.001000, rmse = 0.122480
#public score = 0.12198

# =============================================================================
# NN
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,  MaxoutDense
from keras.regularizers import l1, l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

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
            #print 'rmse loss = %f, val_loss = %f' % (np.sqrt(logs.get('loss')), np.sqrt(logs.get('val_loss')))
            
p = X.shape[1]
x_train = X.values
y_train = y.values

# train到下不去了，再去想Regularization, Dropout, Early Stopping等工具
def nn_model(neurons=10):        
    
    print 'model: p->10->1'
    
    model = Sequential() 
    
    model.add(Dense(neurons, input_dim=p, activation='relu', name='1st hidden layer'))    
        
    model.add(Dense(1, activation='relu', name='output'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

# =============================================================================
# GridSearchCV
#neurons = [10, 50, 100]
#param_grid = dict(neurons=neurons)
#model = KerasRegressor(build_fn=create_model, nb_epoch=100, batch_size=10, validation_split=0.2, verbose=0)
#grid = GridSearchCV(estimator=model, param_grid=param_grid)
#grid_result = grid.fit(x_train, y_train)

#rmses = np.sqrt(grid_result.cv_results_['mean_test_score'])
#params = grid_result.cv_results_['params']
#for rmse, param in zip(rmses, params):
    #print "%f with: %r" % (rmse, param)

# =============================================================================
# ModelCheckpoint
#from keras.callbacks import ModelCheckpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# model 1
neurons = 10
model = nn_model(neurons)
#model.summary()
#model.load_weights("weights.best.hdf5")
history = LossHistory()
hist = model.fit(x_train, y_train, shuffle=True, nb_epoch=100, batch_size=10, validation_split=0.1, callbacks=[history], verbose=0)

str1 = ' rmse = %.6f' % (float(history.losses[len(history.losses)-1]))
str2 = ' val rmse = %.6f' % (float(history.val_losses[len(history.val_losses)-1]))
plt.title(str1 + str2)
plt.plot(history.losses)
plt.plot(history.val_losses)
plt.ylim([0, 0.5])
plt.show()

# =============================================================================   
# cross-validation
estimator = KerasRegressor(build_fn=nn_model, nb_epoch=100, batch_size=10, verbose=0)
rmse_results = np.sqrt(-cross_val_score(estimator, x_train, y_train, scoring="neg_mean_squared_error", cv=10))
print "rmse: %.6f" % (rmse_results.mean())
#nn model
#cv_10, p - 10 - 1, rmse = 0.160480
#public score = 0.16617

# =============================================================================
# 確定cross-validation驗證完model後，把全部的X再拿進去train一次
model = nn_model(neurons)
model.fit(x_train, y_train, shuffle=True, nb_epoch=500, batch_size=10, verbose=0)

# =============================================================================   
# 方法1 :用cross validation 找 a
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

kf = StratifiedKFold(n_splits=10)

for a in np.arange(0, 0.5, 0.1):
    
    kf_rmse = []    
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        kX_train = X.loc[train_index, :]
        ky_train = y.loc[train_index]
        kX_test = X.loc[test_index, :]
        ky_test = y.loc[test_index]
        kModel_lasso = Lasso(0.001)
        kModel_lasso.fit(kX_train, ky_train)
        lasso_pred = kModel_lasso.predict(kX_test)
        
        KModel = nn_model(neurons)
        KModel.fit(kX_train.values, ky_train.values, shuffle=True, nb_epoch=500, batch_size=10, verbose=0)
        nn_pred = KModel.predict(kX_test.values)
        nn_pred = nn_pred[:, 0]
    
        pred = (1-a)*lasso_pred + a*nn_pred
        kf_rmse.append(np.sqrt(mean_squared_error(ky_test, pred)))
        
    print 'a=%f, rmse=%f' % (a, np.mean(kf_rmse))
#a=0.000000, rmse=0.122480
#a=0.100000, rmse=0.121661
#a=0.200000, rmse=0.121098 --> best
#a=0.300000, rmse=0.122300
#a=0.400000, rmse=0.125252

# =============================================================================   
# 方法1 : output
lasso_pred = np.expm1(model_lasso.predict(X_test))
nn_pred = np.expm1(model.predict(X_test.values))
nn_pred = nn_pred[:, 0]

df = pd.DataFrame({"lasso_pred":lasso_pred,"nn_pred":nn_pred})
df.plot(x = "lasso_pred", y = "nn_pred", kind = "scatter")

preds = 0.8*lasso_pred + 0.2*nn_pred

pred_df = pd.DataFrame(preds, index=test["Id"], columns=["SalePrice"])
pred_df.to_csv('D:/data-analysis/output.csv', header=True, index_label='Id')
#public score = 0.12048

# =============================================================================   
# 方法2 :用linear regression 找 a (少了切validation去找a，然後再用validation確認)
from sklearn.linear_model import LinearRegression
lasso_pred = model_lasso.predict(X)
nn_pred = model.predict(X.values)
nn_pred = nn_pred[:, 0]
df = pd.DataFrame({"lasso_pred":lasso_pred,"nn_pred":nn_pred})
print df.corr()
lr = LinearRegression()
lr.fit(df, y)
#lr.coef_
#array([ 0.10905888,  0.89317036])
#lr.intercept_
#-0.046407052933819415

# =============================================================================   
# 方法2:output
lasso_pred = model_lasso.predict(X_test)
nn_pred = model.predict(X_test.values)
nn_pred = nn_pred[:, 0]

df = pd.DataFrame({"lasso_pred":lasso_pred,"nn_pred":nn_pred})
df.plot(x = "lasso_pred", y = "nn_pred", kind = "scatter")
preds = np.expm1(lr.predict(df.values))

pred_df = pd.DataFrame(preds, index=test["Id"], columns=["SalePrice"])
pred_df.to_csv('D:/data-analysis/output.csv', header=True, index_label='Id')
#public score = 0.13788