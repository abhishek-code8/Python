#%matplotlib inline
import matplotlib.pyplot as plt
import math as mt
import numpy as np
import os
import pandas as pd
import seaborn as sn

data= pd.read_excel('breastcancer_dataset_standard_format.xlsx')

data['x6'].replace(-1,np.nan,inplace=True)
sn.heatmap(data.isnull(),cbar=False)
#heatmap to see null values in the dataset
plt.show()

data['x6'].replace(np.nan,0,inplace=True)
mean_col6=data['x6'].mean()
floorcal=mt.floor(mean_col6)
#taking the floor value
data['x6'].replace(0,floorcal,inplace=True)
print(data['x6'].value_counts())
#0 and 1 are catagories
print(data['y'].value_counts())

data.hist(bins=50,figsize  =(15,15))
plt.show()
data.head(5)

data.drop(columns='sample number',inplace=True)

X=data[['x1','x2','x3','x4','x5','x6','x7','x8','x9']]
Y=data['y']



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, y_train)

Y_pred = logisticRegr.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test , Y_pred))
print(Y_pred)