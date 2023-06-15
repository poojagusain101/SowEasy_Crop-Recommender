
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import pickle

df=pd.read_csv("Crop_recommendation.csv")

X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)

LogReg=LogReg.fit(X_train,y_train)
DecisionTree=DecisionTree.fit(X_train,y_train)
NaiveBayes=NaiveBayes.fit(X_train,y_train)
RF=RF.fit(X_train,y_train)

pickle.dump(LogReg.open('LogReg_model.pkl','wb'))
pickle.dump(DecisionTree.open('DecisionTree_model.pkl','wb'))
pickle.dump(NaiveBayes.open('NaiveBayes_model.pkl','wb'))
pickle.dump(RF.open('RF_model.pkl','wb'))
