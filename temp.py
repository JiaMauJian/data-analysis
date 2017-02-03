import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("../data-analysis/house prices/train.csv")
test_df = pd.read_csv("../data-analysis/house prices/test.csv")

# 美國的房子是以平方英尺(square feet)為單位，台灣則是用坪來算房屋的面積。
# 35.58 平方英尺 = 1坪
# 4000平方英尺 = 112坪
#plt.scatter(train_df.GrLivArea, train_df.SalePrice)
#plt.xlabel('GrLivArea')
#plt.ylabel('SalePrice')
#plt.show()
#
#plt.scatter(train_df.TotalBsmtSF, train_df.SalePrice)
#plt.xlabel('TotalBsmtSF')
#plt.ylabel('SalePrice')
#plt.show()

#sns.lmplot('TotalBsmtSF', 'SalePrice', data=train_df, hue='SaleCondition', aspect=2.0, fit_reg=False)

# https://ww2.amstat.org/publications/jse/v19n3/decock/datadocumentation.txt

# drop outlier
train_df.drop(train_df[train_df.GrLivArea > 4000].index, inplace='True')

train_df.drop(train_df[train_df.TotalBsmtSF > 2500].index, inplace='True')
train_df.drop(train_df[train_df.TotalBsmtSF == 0].index, inplace='True')

# replace
train_df.SaleCondition = train_df.SaleCondition.replace(
                        {'Abnorml': 0, 'Alloca': 1, 'AdjLand': 2, 'Family': 3, 'Normal': 4, 'Partial': 5})

test_df.SaleCondition = test_df.SaleCondition.replace(
                        {'Abnorml': 0, 'Alloca': 1, 'AdjLand': 2, 'Family': 3, 'Normal': 4, 'Partial': 5})
# just yes/no
train_df.loc[train_df.Fireplaces > 0, 'Fireplaces']  = 1
test_df.loc[test_df.Fireplaces > 0, 'Fireplaces']  = 1

# 選X參數
x_train = pd.DataFrame()
x_train['GrLivArea'] = train_df.GrLivArea #室內幾坪
x_train['TotalBsmtSF'] = train_df.TotalBsmtSF #總共幾坪
x_train['LotArea'] = train_df.LotArea
x_train['SaleCondition'] = train_df.SaleCondition #屋況
x_train['Fireplaces'] = train_df.Fireplaces #有沒有火爐
x_train['GarageCars'] = train_df.GarageCars #可以停幾台車

# 選y
y_train = pd.DataFrame()
y_train['SalePrice'] = train_df.SalePrice

# model
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.regularizers import l2

p = x_train.shape[1]
x_train = x_train.values
y_train = y_train.values

model = Sequential()
model.add(Dense(input_dim=p, output_dim=30, activation='relu'))
model.add(Dense(output_dim=1, activation='relu'))

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train, nb_epoch=1000, batch_size=10, verbose=0)

#w = model.layers[0].get_weights()[0]
#b = model.layers[0].get_weights()[1]
#w = w.reshape(1, p)
#
#pre_y = np.sum (x_train[0] * w + b) # model.predict(x_train)[0]
#print 'pre_y = %f' % (pre_y)
#
#y = y_train[0]

print 'pre_y - y = %f' % (model.predict(x_train)[0] - y_train[0])

print np.sqrt(model.evaluate(x_train, y_train))

x_test = pd.DataFrame()
x_test['GrLivArea'] = test_df.GrLivArea #室內幾坪
x_test['TotalBsmtSF'] = test_df.TotalBsmtSF #總共幾坪
x_test['LotArea'] = test_df.LotArea
x_test['SaleCondition'] = test_df.SaleCondition #屋況
x_test['Fireplaces'] = test_df.Fireplaces #有沒有火爐
x_test['GarageCars'] = test_df.GarageCars #可以停幾台車
