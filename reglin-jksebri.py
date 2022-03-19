# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd 

///membaca dataset
data = pd.read_csv('../input/jksebri0322/jksebri0322.csv')
data.head()

//print(data.shape)

/// inisiasi dataset
X = data['JKSE'].values
Y = data['BRI'].values


mean_x = np.mean(X)
mean_y = np.mean(Y)

m = len(X)

numeric = 0
denom = 0

for i in range(m):
    numeric +=(X[i] - mean_y) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numeric/denom
b0 = mean_y - (b1 * mean_x)

//print(b1,b0)

/// membuat gambar grafik garis
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

max_x = np.max(X) + 100
min_x = np.min(X) - 100
​
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x
​
plt.plot(x,y, color='#58b970', label = 'Regression Line')
​
plt.scatter(X,Y, color='#ef5423', label = 'Scatter Plot')
​
plt.xlabel('JKSE')
plt.ylabel('BRI')
plt.legend()
//plt.show()

///mencari nilai r2
ss_t = 0
ss_r = 0

for i in range(m):
        y_pred = b0 + b1 * X[i]
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
//print(r2)