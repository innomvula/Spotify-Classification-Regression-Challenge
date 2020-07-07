#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries:
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn import metrics  
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[2]:


#Load Test and Training Data
test_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment\\cs98xspotifyclassification\\CS98XClassificationTest.csv')
train_data = pd.read_csv('C:\\Users\\Inno Mvula\\Desktop\\MSc Quantitative Finance\\S2.CS985 - Machine Learning and Data Analytics\\Assignment\\cs98xspotifyclassification\\CS98XClassificationTrain.csv')


# In[3]:


#Obseravtion of Training Dataset
train_data.info()


# In[4]:


train_data.isnull().sum()


# In[5]:


#Drop rows where genre is missing. Only 15 were missing out of 453 so dropping 15 shouldn't have a huge effect on the dataset
train_data = train_data.dropna()
train_data.info()


# In[6]:


train_data.head()


# In[7]:


#Obseravtion of the distribution of genres
train_data['top genre'].value_counts()


# In[8]:


#Creating the dependent variable class and encoding
factor = pd.factorize(train_data['top genre'])
train_data['top genre'] = factor[0]
definitions = factor[1]
print(train_data['top genre'].head())
print(definitions)


# In[9]:


train_data.head()


# In[10]:


train_data['top genre'].value_counts()


# In[ ]:





# In[11]:


#Drop all genres with only one instance. Synthetic Minority Over-sampling Technique (SMOTE) has issues handling classes with only one instance
counts = train_data['top genre'].value_counts()
rtrain_data = train_data[~train_data['top genre'].isin(counts[counts < 2].index)]
rtrain_data['top genre'].value_counts()


# In[12]:


#reduced dataframe with genres that have 2 or more instances
rtrain_data


# In[ ]:





# In[13]:


#Extracting features and Target
#Splitting the data into independent and dependent variables
X = rtrain_data.iloc[:, 4:14].values
Y = rtrain_data.iloc[:, 14].values
print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(Y[:5])


# In[14]:


len(X), len(Y)


# In[15]:


gen_dict ={}
for gen in rtrain_data['top genre']:
    if gen not in gen_dict:
        gen_dict[gen] = 1
    else:
        gen_dict[gen] += 1
print(gen_dict)


# In[ ]:





# In[16]:


#create sampling strategy dict. This dictionary adds 4 sythentic duplicates to each class with less than 60 instances.
#Reason for choosing 4 is because after experimenting with values, adding 4 instances gave me the best results
#the dataset is still relatively imablanced but there is slight improvement.
for key, value in gen_dict.items():
    if value < 60:
        gen_dict[key] += 4
print(gen_dict)


# In[ ]:





# In[17]:


#improving the balance of the dataset using SMOTE.
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 42, k_neighbors = 1, sampling_strategy = gen_dict )
x_train_res, y_train_res = sm.fit_sample(X, Y)


# In[18]:


len(x_train_res), len(y_train_res)


# In[19]:


#Creating the Training and Test set from data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train_res, y_train_res, test_size = 0.10, random_state = 42, stratify = y_train_res)


# In[ ]:





# In[20]:


len(X_train)


# In[21]:


import collections
print(collections.Counter(Y_train))


# In[22]:


len(collections.Counter(Y_train))


# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


#Here we have selected several parameters we would like to tune to get optimal values in each that improve our model
#parameters = {"n_estimators":num_est, "max_depth": max_dep, "criterion": crit, "max_features": max_feat}
rfc = RandomForestClassifier(random_state = 42)
num_est = []
max_dep = []
crit = ['gini', 'entropy']
max_feat = []
for i in range(5, 100, 5):
    num_est.append(i)
for l in range(2, 16, 2):
    max_dep.append(l)
max_dep.append(None)
for f in range(1, 11, 1):
    max_feat.append(f)
parameters = {
    "n_estimators":num_est,
    "max_depth": max_dep,
    "criterion": crit,
    "max_features": max_feat
    
}
print(parameters)


# In[24]:


#Here we use GridSearch to hypertune our parameters to get optimal values. Gridsearch iterates through all the values
#set for each parameter and returns the best ones
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfc,parameters,cv = 3)
cv.fit(X_train, Y_train)


# In[38]:


#Function used to print out different parameter combinations and their scores
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score, std_score, params):
        if round(mean,3) >= 0.57:
            print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[39]:


display(cv)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


#Train and fit our model using the recommended values for our parameters
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 80, random_state = 42, criterion = 'entropy', max_features = 3, max_depth = 14)
ranfor.fit(X_train, Y_train)


# In[41]:


#evaluation
Y_pred = ranfor.predict(X_test)
#Reverse factorize
reversefactor = dict(zip(range(86),definitions))
Y_test = np.vectorize(reversefactor.get)(Y_test)
Y_pred = np.vectorize(reversefactor.get)(Y_pred)
# Making the Confusion Matrix
print(pd.crosstab(Y_test, Y_pred, rownames=['Actual genres'], colnames=['Predicted genres']))


# In[42]:


#print a classification report depicting the precision, recall, and f1-score of t=the different classes and overall model
from sklearn.metrics import classification_report
class_rep_forest = classification_report(Y_test, Y_pred)
print(class_rep_forest)


# In[43]:


#Storing the trained model
#We are going to observe the importance for each of the features and then store the Random Forest classifier using the joblib function of sklearn.
print(list(zip(train_data.columns[4:14], ranfor.feature_importances_)))


# In[ ]:





# In[44]:


# predicting test data results
test_data.head()


# In[45]:


#Extracting our features from the test data and predicting their genres
X1 = test_data.iloc[:, 4:14].values
reversefactor = dict(zip(range(86),definitions))
test_data['top genre'] = ranfor.predict(X1)
test_data['top genre'] = np.vectorize(reversefactor.get)(test_data['top genre'])


# In[46]:


test_data.head(10)


# In[47]:


test_data['top genre'].value_counts()


# In[48]:


#saving predictions as a csv for submission on kaggle
prediction = test_data[['Id', 'top genre']]
prediction.to_csv("SamplingRF29_submission.csv", index=False)
prediction.tail(10)


# In[ ]:




