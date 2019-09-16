import pandas as pd 
import datetime
import pandas_datareader.data as web
from pandas import Series ,DataFrame

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

start=datetime.datetime(2015,1,1)
end=datetime.datetime(2019,9,15)

# huawei=web.DataReader("002502.SZ",'yahoo',start,end)
hw=pd.read_csv('hw.csv')
hw.set_index('Date',inplace=True)
# print(hw.head())
#plot the adj close 
hw['Adj Close'].plot(label='huawei', legend=True)
# plt.show()
plt.clf()
#hist of daily price change
hw['Adj Close'].pct_change().plot.hist(bins=50)
plt.xlabel('Adj Close')
# plt.show()

# Here i will calculate the average
df=hw.copy()
df['Average'] = (df['Open'] + df['Close'] + df['High'] + df['Low']) / 4
print(df.head())

frc=30 
df['Future Close'] = df[['Adj Close']].shift(-frc)
print(df.head())
#calculate corellation Matrix
corr = df[['Average', 'Future Close']].corr()
print(corr)

#ml part
#preprocessing 
# Drop missing values from data frame hw

# Next step : i want only the Adj close column so drop the rest
drop_column=['Open','High','Low','Close','Adj Close','Volume','Future Close']

df.dropna(inplace=True)
x=df.drop(drop_column,axis=1)
print(x)
#scale all the values
x=preprocessing.scale(x)
y=df['Future Close']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(y_train)

#linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)

# KNN Regression
knnreg = KNeighborsRegressor(n_neighbors=2)
knnreg.fit(x_train, y_train)

# Quadratic Regression 2
qrpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
qrpoly2.fit(x_train, y_train)



lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence * 100,'%')

knnreg_confidence = knnreg.score(x_test, y_test)
print("knn confidence: ", knnreg_confidence * 100,'%')
qrpoly2_confidence = qrpoly2.score(x_test, y_test)
print("qrpoly2 confidence: ", qrpoly2_confidence * 100,'%')




