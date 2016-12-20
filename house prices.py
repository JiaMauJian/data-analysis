import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

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
        rmse = np.sqrt(-cross_val_score(Lasso(alpha=10**alpha), X, y, scoring="neg_mean_squared_error", cv = 5))
        print 'alpha = %f, rmse = %f' % (10**alpha, rmse.mean())    
        alphas.append(10**alpha)
        rmse_list.append(rmse.mean())
    return rmse_list, alphas

rmse, alphas = rmse_cv(X, y)

cv_lasso = pd.Series(rmse, index=alphas)
cv_lasso.plot(logx=True)

# 用內建的function
model_lasso_cv = LassoCV(alphas=alphas)
model_lasso_cv.fit(X, y)
model_lasso_cv.score(X, y)

# Lasso(alpha=0.001) 
# alpha = 0.001000, rmse = 0.123366
# public score = 0.15588

# =============================================================================
# NN
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l1, l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(np.sqrt(logs.get('loss')))
        self.val_losses.append(np.sqrt(logs.get('val_loss')))
        print 'rmse loss = %f, val_loss = %f' % (np.sqrt(logs.get('loss')), np.sqrt(logs.get('val_loss')))       
        #print 'rmse loss = %f' % (np.sqrt(logs.get('loss')))
        
p = X.shape[1]
x_train = X.values
y_train = y.values

# train到下不去了，再去想Regularization, Dropout, Early Stopping等工具
def create_model(neurons=10):        
    model = Sequential() 
    model.add(Dense(input_dim=p, output_dim=neurons))
    model.add(Activation('relu'))    
        
    model.add(Dense(output_dim=1))
    model.add(Activation('relu'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model

# 金字塔型
def create_model_pyramid(layers, dropout):
    
    model = Sequential()
    
    if layers > 0:        
        model.add(Dense(input_dim=p, output_dim=p*3))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    
    if layers > 1:    
        model.add(Dense(output_dim=p*2))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    if layers > 2:
        model.add(Dense(output_dim=p))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    
    model.add(Dense(output_dim=1))
    model.add(Activation('relu'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model
    
neurons = [10, 50, 100]
param_grid = dict(neurons=neurons)
model = KerasRegressor(build_fn=create_model, nb_epoch=100, batch_size=10, validation_split=0.2, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)

rmses = np.sqrt(grid_result.cv_results_['mean_test_score'])
params = grid_result.cv_results_['params']
for rmse, param in zip(rmses, params):
    print "%f with: %r" % (rmse, param)

from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# model 1
neurons = 128
model = create_model_pyramid(3, neurons, 0)
#model.load_weights("weights.best.hdf5")
history = LossHistory()
hist = model.fit(x_train, y_train, nb_epoch=500, batch_size=10, validation_split=0.2, callbacks=[history, checkpoint], verbose=2)
str1 = ' loss = %.6f' % (float(history.losses[len(history.losses)-1]))
str2 = ' val loss = %.6f' % (float(history.val_losses[len(history.val_losses)-1]))
#plt.title('dropout 0.1, 3 hidden ' + str(neurons) +  ' neurons ' + str1 + str2)
plt.title('dropout 0.1, 1st hidden 384, 2nd hidden 256, 3rd 128 ' + str1 + str2)
plt.plot(history.losses)
plt.plot(history.val_losses)
plt.ylim([0, 0.5])
plt.show()

# model 2
model = create_model2(1, 2, 0)
history = LossHistory()
hist = model.fit(x_train, y_train, nb_epoch=300, batch_size=10, validation_split=0.2, callbacks=[history], verbose=0)
plt.plot(history.losses)
plt.plot(history.val_losses)
plt.ylim([0, 0.5])
plt.show()

# =============================================================================   
# output
lasso_pred = np.expm1(model_lasso_cv.predict(X_test))
dnn_pred = np.expm1(model.predict(X_test.values))
dnn_pred = dnn_pred[:, 0]

df = pd.DataFrame({"lasso_pred":lasso_pred,"dnn_pred":dnn_pred})
df.plot(x = "lasso_pred", y = "dnn_pred", kind = "scatter")

preds = 0.7*lasso_pred + 0.3*dnn_pred

pred_df = pd.DataFrame(preds, index=test["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')