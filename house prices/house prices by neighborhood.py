import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

train  = pd.read_csv("D:/data-analysis/house prices/train.csv")
test = pd.read_csv("D:/data-analysis/house prices/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

neighborhood_map = {
        "MeadowV" : 1,  #  88000
        "IDOTRR" : 1,   # 103000
        "BrDale" : 1,   # 106000
        "OldTown" : 1,  # 119000
        "Edwards" : 1,  # 119500
        "BrkSide" : 1,  # 124300
        "Sawyer" : 1,   # 135000
        "Blueste" : 1,  # 137500
        "SWISU" : 2,    # 139500
        "NAmes" : 2,    # 140000
        "NPkVill" : 2,  # 146000
        "Mitchel" : 2,  # 153500
        "SawyerW" : 2,  # 179900
        "Gilbert" : 2,  # 181000
        "NWAmes" : 2,   # 182900
        "Blmngtn" : 2,  # 191000
        "CollgCr" : 2,  # 197200
        "ClearCr" : 3,  # 200250
        "Crawfor" : 3,  # 200624
        "Veenker" : 3,  # 218000
        "Somerst" : 3,  # 225500
        "Timber" : 3,   # 228475
        "StoneBr" : 4,  # 278000
        "NoRidge" : 4,  # 290000
        "NridgHt" : 4,  # 315000
    }

train["NeighborhoodBin"] = train["Neighborhood"].map(neighborhood_map)
test["NeighborhoodBin"] = test["Neighborhood"].map(neighborhood_map)

all_data["NeighborhoodBin"] = all_data["Neighborhood"].map(neighborhood_map)
all_data["Model"] = ''

all_data.loc[:, "Model"].loc[all_data["NeighborhoodBin"]==1] = '1'
all_data.loc[:, "Model"].loc[all_data["NeighborhoodBin"]==2] = '2'
all_data.loc[:, "Model"].loc[all_data["NeighborhoodBin"]==3] = '3'
all_data.loc[:, "Model"].loc[all_data["NeighborhoodBin"]==4] = '4'

# =============================================================================
# Data Preprocessing
def data_prerpocessing(data):    
    numeric_feats = data.dtypes[data.dtypes != "object"].index
    
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    
    data[skewed_feats] = np.log1p(data[skewed_feats])
    
    data = pd.get_dummies(data)
    
    data = data.fillna(data.median())
            
    scaler = StandardScaler().fit(data[numeric_feats])
    data[numeric_feats] = scaler.transform(data[numeric_feats])
    
    return data

all_data = data_prerpocessing(all_data)

# 打all_data.loc[:5, :]取前5筆資料，為什麼會出現錯誤
# KeyError: 'Cannot get right slice bound for non-unique label: 5'
# 因為index的type很不一樣，
print train.index
print all_data.index

# 做一下reset
all_data = all_data.reset_index(drop=True)


#creating matrices for sklearn:
X1 = all_data.loc[:train.shape[0]-1, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_1'] == 1]
X_test1 = all_data.loc[train.shape[0]:, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_1'] == 1]
y1 = train['SalePrice'].loc[train["NeighborhoodBin"] == 1]
y1 = np.log1p(y1)
print X1.shape[0], y1.shape

X2 = all_data.loc[:train.shape[0]-1, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_2'] == 1]
X_test2 = all_data.loc[train.shape[0]:, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_2'] == 1]
y2 = train['SalePrice'].loc[train["NeighborhoodBin"] == 2]
y2 = np.log1p(y2)
print X2.shape[0], y2.shape

X3 = all_data.loc[:train.shape[0]-1, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_3'] == 1]
X_test3 = all_data.loc[train.shape[0]:, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_3'] == 1]
y3 = train['SalePrice'].loc[train["NeighborhoodBin"] == 3]
y3 = np.log1p(y3)
print X3.shape[0], y3.shape

X4 = all_data.loc[:train.shape[0]-1, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_4'] == 1]
X_test4 = all_data.loc[train.shape[0]:, 'MSSubClass':'SaleCondition_Partial'].loc[all_data['Model_4'] == 1]
y4 = train['SalePrice'].loc[train["NeighborhoodBin"] == 4]
y4 = np.log1p(y4)
print X4.shape[0], y4.shape

# =============================================================================
# Lasso
#from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import Lasso, LassoCV
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

rmse, alphas = rmse_cv(X1, y1)
#alpha = 0.010000, rmse = 0.176245
rmse, alphas = rmse_cv(X2, y2)
#alpha = 0.001000, rmse = 0.090381
rmse, alphas = rmse_cv(X3, y3)
#alpha = 0.010000, rmse = 0.127674
rmse, alphas = rmse_cv(X4, y4)
#alpha = 0.010000, rmse = 0.117354

# =============================================================================
# 確定cross-validation驗證完model後，把全部的X再拿進去train一次
pred = pd.DataFrame(0, index=test['Id'], columns=["SalePrice"])

model_lasso1 = Lasso(0.010000)
model_lasso1.fit(X1, y1)
pred1 = np.expm1(model_lasso1.predict(X_test1))
pred.loc[pred.index[test["NeighborhoodBin"] == 1]] = pred1.reshape(pred1.shape[0], 1)
pred = pred.reset_index(drop=True)

model_lasso2 = Lasso(0.001000)
model_lasso2.fit(X2, y2)
pred2 = np.expm1(model_lasso2.predict(X_test2))
pred.loc[pred.index[test["NeighborhoodBin"] == 2]] = pred2.reshape(pred2.shape[0], 1)

model_lasso3 = Lasso(0.010000)
model_lasso3.fit(X3, y3)
pred3 = np.expm1(model_lasso3.predict(X_test3))
pred.loc[pred.index[test["NeighborhoodBin"] == 3]] = pred3.reshape(pred3.shape[0], 1)

model_lasso4 = Lasso(0.010000)
model_lasso4.fit(X4, y4)
pred4 = np.expm1(model_lasso4.predict(X_test4))
pred.loc[pred.index[test["NeighborhoodBin"] == 4]] = pred4.reshape(pred4.shape[0], 1)


# 有分neighorhood建model，最後分數是 
# Your submission scored 0.12275
# 比沒有分的分數 = 0.12198，還差，崩潰
pred.to_csv('output.csv', header=True, index_label='Id')