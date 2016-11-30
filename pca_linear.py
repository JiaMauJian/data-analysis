# http://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
# 了解PCA是什麼東西

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

x1 = pd.Series(np.linspace(0, 5.0, 10))
x2 = pd.Series(x1 + np.random.randn(10) * 0.33)

plt.scatter(x1, x2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# loading data
df = pd.DataFrame({'x1':x1, 'x2':x2})

# feature scaling
scaler = StandardScaler().fit(df)
df_scaled = scaler.transform(df)

print 'mean = %f' % (df_scaled[:, 0].mean())
print 'std = %f' % (df_scaled[:, 0].std())

# pca
pca = PCA(n_components=2)
pca.fit(df_scaled)

print 'variance(eigenvalue) ratio = %s ' % (pca.explained_variance_ratio_)

print 'eigenvectors'
print pca.components_

print 'covariance matrix'
print pca.get_covariance()

print 'the variance of x1 = %f' %  (pca.get_covariance()[0][0])

print 'the variance of x2 = %f' %  (pca.get_covariance()[1][1])

print 'the covariance between them = %f' %  (pca.get_covariance()[0][1])

print 'PC1 weights = %s' % (pca.components_[0])

print 'PC2 weights = %s' % (pca.components_[1])

# PC的值怎麼算 (z = w1x1 + w2x2)
df_z_PC1 = pd.DataFrame(np.sum(df_scaled*pca.components_[0], axis=1), columns=['PC1'])
df_z_PC2 = pd.DataFrame(np.sum(df_scaled*pca.components_[1], axis=1), columns=['PC2'])

# PC的值的分布圖
plt.scatter(df_z_PC1, df_z_PC2)
plt.xlabel('PC1')
plt.xlim([-3, 3])
plt.ylabel('PC2')
plt.ylim([-3, 3])
plt.show()

# 同上
#pcscores = pd.DataFrame(pca.transform(df_scaled))
#pcscores.columns = ['PC'+str(i+1) for i in range(len(pcscores.columns))]