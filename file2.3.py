#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:41:08 2019

@author: pranitha
"""

import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
df=pd.read_csv('/home/pranitha/internship/airlinesdata.csv')

df.isnull().any().any()
df.isnull().any()
df.isnull().sum()
data=df.dropna(how='any')
data.shape
data.isnull().any().any()

df['Gender'].value_counts()
data.info()
churn=data.groupby('Churn')
churn.mean()

data.describe()

sizes = data['Churn'].value_counts(sort = True)
colors = ["grey","purple"] 
rcParams['figure.figsize'] = 5,5

# Plot
plt.pie(sizes,labels=['churn','non-churn'],colors=colors,autopct='%1.1f%%', shadow=True, startangle=270)

plt.title('Percentage of Churn in Dataset')
plt.show()
fdata=data.drop(['id','satisfaction_v2','Customer Type'],axis=1)
fdata.info()#19957 rows 22 columns
data.info()#19957 rows 25 columns
fdata=pd.get_dummies(fdata)
fdata.info()#19957 rows 26 columns

#splitting data
X =  fdata.drop(['Churn'], axis=1)
y = fdata['Churn']
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
X_train.info()#15965 rows 25 columns
X_test.info()#3992 rows 25 columns
len(y_train)#15965
len(y_test)#3992

#logistic regression model building
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)
#logistic regression model testing
from sklearn import metrics
predictions= model.predict(X_test)
pr=pd.Series(predictions)
#pie plot
plt.pie(pr.value_counts(sort='True'),labels=['churn','non-churn'],colors=colors,autopct='%1.1f%%', shadow=True, startangle=270)

plt.title('Percentage of Churn in Dataset')
plt.show()#non-churners 49.3% churners 50.7% in test data
#confusion matrix
import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y_test,predictions))
# Print the prediction accuracy
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))#gives precision,recall,f1-score etc
print(accuracy_score(y_test, predictions ))#gives acuracy of the model #0.824

#now checking the weight of each variable
weights = pd.Series(model.coef_[0],index=X.columns.values)
weights.sort_values(ascending = False)
#roc curve and auc score
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#Area under curve score is 0.898
plt.legend(loc=4)
plt.show()

#random forest algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=0)  
model.fit(X_train, y_train)  
#random forest model testing
from sklearn import metrics
predictions= model.predict(X_test)
pr=pd.Series(predictions)
#pie plot
plt.pie(pr.value_counts(sort='True'),labels=['churn','non-churn'],colors=colors,autopct='%1.1f%%', shadow=True, startangle=270)

plt.title('Percentage of Churn in Dataset')
plt.show()#non-churners 49.2% churners 50.8% in test data
#confusion matrix
import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y_test,predictions))
# Print the prediction accuracy
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))#gives precision,recall,f1-score etc
print(accuracy_score(y_test, predictions ))#gives acuracy of the model #0.945
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
#roc curve and auc score
y_pred_proba = model.predict_proba(X_test)[::,1]#can use model.predict fucntion too.gives 0 or 1, thus curve is sharp
#model.predict_proba gives a range of prabability of occurence of 0 and 1. This gives a smooth roc curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#area under curve score is 0.989
plt.legend(loc=4)
plt.show()


#decision tree algorithm
from sklearn import tree
# Initialize our decision tree object
model = tree.DecisionTreeClassifier(max_depth=4)#4 levels
# Train our decision tree (tree induction + pruning)
model.fit(X_train, y_train)

#visualising a tree
with open("/home/pranitha/internship/class.txt","w") as f:
    f=tree.export_graphviz(model,out_file=f,feature_names=list(X.columns))
#go to class.txt file and copy paste the content into webgraphviz.com
#testing decision tree
from sklearn import metrics
predictions= model.predict(X_test)
pr=pd.Series(predictions)
#pie plot
plt.pie(pr.value_counts(sort='True'),labels=['churn','non-churn'],colors=colors,autopct='%1.1f%%', shadow=True, startangle=270)

plt.title('Percentage of Churn in Dataset')
plt.show()#non-churners 49.2% churners 50.8% in test data
#confusion matrix
import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y_test,predictions))
# Print the prediction accuracy
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions ))#gives precision,recall,f1-score etc
print(accuracy_score(y_test, predictions ))#gives acuracy of the model #0.859
#roc curve and auc score
y_pred_proba = model.predict_proba(X_test)[::,1]#can use model.predict fucntion too.gives 0 or 1, thus curve is sharp
#model.predict_proba gives a range of prabability of occurence of 0 and 1. This gives a smooth roc curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#area under curve score is 0.912
plt.legend(loc=4)
plt.show()

