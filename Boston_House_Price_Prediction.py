#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

import warnings


# In[2]:


dataset = pd.read_csv('HousingData.csv')


# In[3]:


dataset.drop_duplicates()


# In[4]:


dataset.isna().sum()


# In[5]:


dataset['CRIM'].fillna(dataset['CRIM'].mean(), inplace = True)
dataset['ZN'].fillna(dataset['ZN'].mean(), inplace = True)
dataset['INDUS'].fillna(dataset['INDUS'].mean(), inplace = True)
dataset['CHAS'].fillna(dataset['CHAS'].mean(), inplace = True)
dataset['AGE'].fillna(dataset['AGE'].mean(), inplace = True)
dataset['LSTAT'].fillna(dataset['LSTAT'].mean(), inplace = True)
dataset.isna().sum()


# In[6]:


dataset.info()
print('-'*50)
dataset.describe()


# In[ ]:





# In[7]:


X = dataset.drop('MEDV',1)
y = dataset['MEDV']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[9]:


X_train_cv, X_cv ,y_train_cv, y_cv = train_test_split(X_test,y_test,test_size = 0.50)


# In[10]:


print(X_train_cv.shape,y_train_cv.shape)
print(X_cv.shape,y_cv.shape)


# In[ ]:





# In[11]:


model = LinearRegression()


# In[12]:


model.fit(X_train,y_train)


# In[ ]:





# In[13]:


prediction = model.predict(X_test)
print(mean_absolute_error(y_test,prediction))


# In[14]:


#plt.scatter(y_test,prediction)
#plt.xlabel('y_test')
#plt.xlabel('prediction')


# In[20]:


# Test Options and Evaluation Metrics
num_folds = 10
scoring = "neg_mean_squared_error"
seed = 51
# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),   cv_results.std())
    print(msg)


# In[16]:


#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()


# In[25]:


# SVR on Train set

'''scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
# Build parameter grid
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
# Build the model
model = SVR()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)
# Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))'''


# In[27]:


# KNN

'''scaler = StandardScaler().fit(X_train_cv)
rescaledX = scaler.transform(X_train_cv)
# Build parameter grid
n_neighbour = np.arange(0,20)
param_grid = dict(n_neighbors = n_neighbour)
# Build the model
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train_cv)
# Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) '''


# In[26]:


# SVR on Validation set

'''scaler = StandardScaler().fit(X_train_cv)
rescaledX = scaler.transform(X_train_cv)
# Build parameter grid
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
# Build the model
model = SVR()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train_cv)
# Show the results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) '''


# In[28]:


model = SVR(C=0.5,kernel='linear')


# In[29]:


model.fit(X_train,y_train)


# In[30]:


#print(mean_absolute_error(y_test,model.predict(X_test)))


# In[31]:


import pickle


# In[33]:


pickle.dump(model, open('SVR_Boston_house_prive_prediction.pkl','wb'))


# In[ ]:




