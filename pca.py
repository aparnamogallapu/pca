# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 08:35:11 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\u_datasets\\PCA\\Wine.csv")
data=dataset.copy()

dataset.shape
dataset.columns

x=dataset.iloc[:,0:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
explained_variance=pca.explained_variance_ratio_


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
classifier.score(x_train,y_train)
classifier.fit(x_test,y_test)
classifier.score(x_test,y_test)


y_pred=classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)






