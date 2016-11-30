# https://www.kaggle.com/strakul5/d/abcsds/pokemon/principal-component-analysis-of-pokemon-data/notebook
# http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/dim%20reduction%20(v5).pdf
# http://140.112.21.35:2880/~tlkagk/pokemon/pca.html

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# loading data
df = pd.read_csv('Pokemon.csv')

# renaming one column
columns = df.columns.tolist()
columns[0] = 'id'
df.columns = columns

df.head()

# feature transform
cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
scaler = StandardScaler().fit(df[cols])
df_scaled = scaler.transform(df[cols])

print df_scaled[:, 0].mean()
print df_scaled[:, 0].std()

# pca
pca = PCA(n_components=6)
pca.fit(df_scaled)

# 第5和6的作用比較小，project出來的variance是很小的，選前四個就可以
print 'eigenvalue ratio = %s ' % (pca.explained_variance_ratio_)

pcscores = pd.DataFrame(pca.transform(df_scaled))
pcscores.columns = ['PC'+str(i+1) for i in range(len(pcscores.columns))]

loadings = pd.DataFrame(pca.components_, columns=cols)
loadings.index = ['PC'+str(i+1) for i in range(len(pcscores.columns))]
                  
ax = sns.heatmap(loadings.loc['PC1':'PC4', :].transpose(), center=0, linewidths=0.5, cmap="RdBu", vmin=-1, vmax=1, annot=True)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)

plt.figure()

plt.scatter(pcscores.PC1, pcscores.PC2)
plt.scatter(pcscores.PC1[pcscores.PC1>4], pcscores.PC2[pcscores.PC1>4], color='red') #PC1大於4
plt.scatter(pcscores.PC1[pcscores.PC2>6], pcscores.PC2[pcscores.PC2>6], color='red') #PC2大於6
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.scatter(pcscores.PC3, pcscores.PC4)
plt.scatter(pcscores.PC3[pcscores.PC3>4], pcscores.PC4[pcscores.PC3>4], color='red') #PC3大於4
plt.scatter(pcscores.PC3[pcscores.PC4>6], pcscores.PC4[pcscores.PC4>6], color='red') #PC4大於6
plt.xlabel('PC3')
plt.ylabel('PC4')
plt.show()

# 前2名
print 'PC4 Top 2'
best = pcscores.sort_values(by='PC4', ascending=False)[:2]
print df.loc[best.index]

# 前1名
print 'PC3 Top 1'
best = pcscores.sort_values(by='PC3', ascending=False)[:1]
print df.loc[best.index]