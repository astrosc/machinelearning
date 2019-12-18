# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:43:06 2019

Gaussian Naive Bayes: GaussianNB()
p.351 Python Data Science Handbook

@author: astrosc
"""


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)

model=GaussianNB()
model.fit(Xtrain, ytrain)

# classification report
from sklearn.metrics import f1_score, classification_report
predictions=model.predict(Xtest)
print(classification_report(ytest, predictions))   #helpful fct

from sklearn.metrics import accuracy_score 
ypred = model.predict(Xtest)
print('Misclassified samples: %d' % (ytest != ypred).sum())
print('Accuracy: %.2f' % accuracy_score(ytest, ypred))

"""
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       1.00      0.94      0.97        16
           2       0.90      1.00      0.95         9

    accuracy                           0.97        38
   macro avg       0.97      0.98      0.97        38
weighted avg       0.98      0.97      0.97        38

outperforms logistic regression!!!
"""