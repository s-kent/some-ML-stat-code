
# this python code constructs 2 binary classifier that made top 10(out of 34) in a kaggle in class competition("sinkhole or not")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier , VotingClassifierl, BaggingClassifier , AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import preprocessing


df = pd.read_csv('train.csv' )
df2 = pd.read_csv('test.csv')

X = df.iloc[:,1:11].copy()
y = df.iloc[:,11:].copy()
X = np.array(X)
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
       
 
X = pt.fit_transform(X) 


y= np.array(y)
print(y.shape,X.shape)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.18, random_state=0)

X2 = df2.iloc[:,1:11].copy()
X2 = np.array(X2)
X2 = pt.fit_transform(X2)


clf = RandomForestClassifier(n_estimators=100, max_depth=18,random_state=0,min_impurity_decrease=0.0002,criterion='entropy')



bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME.R", n_estimators=200)
bdt.fit(X,y)
print("ada score-train :",bdt.score(X,y))
print("ada score-test :",bdt.score(X_test,y_test))
bagging = BaggingClassifier(bdt, max_samples=2500, max_features=X.shape[1],bootstrap_features=False,n_estimators=20)
bdt = bagging.fit(X,y)
print("ada_bag score-train :",bdt.score(X,y))
print("ada_bag score-test :",bdt.score(X_test,y_test))


clf5 = ensemble.GradientBoostingClassifier(n_estimators=900,max_depth=4)


clf7 = VotingClassifier(estimators=[('XGboost',clf5),('ada_bags', bdt), ('randomfor', clf)], voting='soft', weights=[1,2,1])
clf7 = clf7.fit(X,y)
print("voting score-train :",clf7.score(X,y))
print("voting score-test :",clf7.score(X_test,y_test))




pre = clf7.predict(X2)
pre2 = bdt.predict(X2)
print(df2.shape,pre.shape)
dic1 = { 'pred': pre }
dic2 = {'pred2':pre2 }
dic1 = pd.DataFrame(dic1)
dic2 = pd.DataFrame(dic2)


dic1.to_csv('out1.csv')
dic2.to_csv('out2.csv')




