# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:32:38 2019

Classificator comparison

@author: astrosc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

X,y=make_moons(n_samples=120, noise=0.3)
X = StandardScaler().fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, random_state=1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


names=['Gaussian Naive Bayes','Logistic Regression','LDA', 'K-Nearest Neighbors', 
       'Random Forest', 'Neural Net', 'Support Vector (linear)', 'Support Vector (rbf)',
       'Gradient Boost', 'XGBClassifier']
       
classification=[GaussianNB(),
                LogisticRegression(random_state=0),
                LinearDiscriminantAnalysis(),
                KNeighborsClassifier(n_neighbors=7, p=2, metric='minkowski'),
                RandomForestClassifier(max_depth=2, n_estimators=5),
                MLPClassifier(max_iter=1000),
                SVC(kernel="linear", probability=1),
                SVC(kernel="rbf", probability=1),
                GradientBoostingClassifier(),
                XGBClassifier()]

fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(3, 4, 1)
plt.scatter(X[y==0,0], X[y==0,1], c='lightblue', edgecolors='k')
plt.scatter(X[y==1,0], X[y==1,1], c='r', edgecolors='k')
ax.set_title("Input data")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

i=1
fpr=dict()
tpr=dict()
ras=[]

for name, model in zip(names, classification):
    model.fit(Xtrain, ytrain)
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    ax = plt.subplot(3, 4, i+1)
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap='bwr', alpha=.8)
    plt.scatter(Xtest[ytest==0,0], Xtest[ytest==0,1], c='lightblue', edgecolors='k')
    plt.scatter(Xtest[ytest==1,0], Xtest[ytest==1,1], c='r', edgecolors='k')
    ax.set_title(name)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    handle=ax.text(x_min+0.5, y_max-0.5, 'acc: %.2f' % model.score(Xtest, ytest), size=10, color='r')

    pred=model.predict_proba(Xtest)[:, 1]
    fpr[i-1], tpr[i-1], _ = roc_curve(ytest, pred)

    pred2 = model.predict(Xtest)    
    ras.append(roc_auc_score(y_true=ytest, y_score=pred2))
    
    i+=1
plt.tight_layout()

#plot ROC-AUC-values of classifiers

ax = plt.subplot(3, 4, 12)
y_pos = np.arange(len(ras))
ax.barh(y_pos, ras, align='center')
ax.set_yticks(y_pos)
ax.tick_params(axis="y", direction="in", pad=-100, labelsize=8, color='w')
ax.tick_params(axis="x", labelsize=8)
ax.set_yticklabels(names, color='w')
ax.set_xlabel('Area under the curve', fontsize=8)
ax.set_title('ROC-AUC')
plt.tight_layout()
