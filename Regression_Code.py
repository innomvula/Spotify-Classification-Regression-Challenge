#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing important packages
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[3]:


#Importing preprocessing and regression packages
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


# In[5]:


#Defining the training and testing dataset
train_data = pd.read_csv(r"C:\Users\Greig\Documents\Uni\5th year\CS 986\Assignment\CS98XRegressionTrain.csv")
test_data = pd.read_csv(r"C:\Users\Greig\Documents\Uni\5th year\CS 986\Assignment\CS98XRegressionTest.csv")


# In[6]:


#Removing NaN values found in the Top Genre column
train_data = train_data.dropna(subset=['top genre'])


# In[7]:


#Converting column to string to allow encoding
train_data['top genre'] = train_data['top genre'].astype(str)
test_data['top genre'] = test_data['top genre'].astype(str)


# In[8]:


#Encoding the Train and Test Data
le = preprocessing.LabelEncoder()
train_data['title'] = le.fit_transform(train_data['title'])
test_data['title'] = le.fit_transform(test_data['title'])
train_data['artist'] = le.fit_transform(train_data['artist'])
test_data['artist'] = le.fit_transform(test_data['artist'])
train_data['top genre'] = le.fit_transform(train_data['top genre'])
test_data['top genre'] = le.fit_transform(test_data['top genre'])


# In[9]:


#Creating a 'Beats' column based on the Beats per Minute and Duration of the song
for dataset in train_data:
    train_data['beats'] = train_data['bpm'] * train_data['dur']/60
for dataset in train_data:
    test_data['beats'] = test_data['bpm'] * test_data['dur']/60


# In[10]:


#Defining Training and Testing Variables
x_train = train_data[['title','artist','top genre','year','bpm','nrgy','dnce','val','dur','acous','spch','beats']]
x_test = test_data[['title','artist','top genre','year','bpm','nrgy','dnce','val','dur','acous','spch','beats']]
y_train = train_data[['pop']]


# In[11]:


#Defining a Random Forest Regression function for later iterated use
def fit_forest_reg(X,Y):
    #Fit random forest regression model and return RSS and R squared values
    model_k = RandomForestRegressor(n_estimators=10,max_depth=4,random_state=0)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared


# In[12]:


#Importing tqdm for the progress bar
from tqdm import tnrange

#Initialization variables for iterated Random Forest Regression (redefining feature and target variables)
X = train_data[['title','artist','top genre','year','bpm','nrgy','dnce','val','dur','acous','spch','beats']]
Y = train_data[['pop']]
k = 12
RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []

#Looping over k = 1 to k = 12 features in X
for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):

    #Looping over all possible combinations: from 12 choose k
    for combo in itertools.combinations(X.columns,k):
        tmp_result = fit_forest_reg(X[list(combo)],Y)   #Store temp result 
        RSS_list.append(tmp_result[0])                  #Append lists
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo)) 


# In[13]:


#Store in DataFrame
df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})
df_min = df[df.groupby('numb_features')['RSS'].transform(min) == df['RSS']]
df_max = df[df.groupby('numb_features')['R_squared'].transform(max) == df['R_squared']]


# In[14]:


#Defining the minimum RSS and maximum R2 value for graphing purposes
df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)


# In[15]:


#Visualising the minimum RSS and maximum R2 values
fig = plt.figure(figsize = (16,6))
ax = fig.add_subplot(1, 2, 1)

ax.scatter(df.numb_features,df.RSS, alpha = .2, color = 'darkblue' )
ax.set_xlabel('# Features')
ax.set_ylabel('RSS')
ax.set_title('RSS - Best subset selection')
ax.plot(df.numb_features,df.min_RSS,color = 'r', label = 'Best subset')
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.scatter(df.numb_features,df.R_squared, alpha = .2, color = 'darkblue' )
ax.plot(df.numb_features,df.max_R_squared,color = 'r', label = 'Best subset')
ax.set_xlabel('# Features')
ax.set_ylabel('R squared')
ax.set_title('R_squared - Best subset selection')
ax.legend()

plt.show()


# In[16]:


#Forward Stepwise Selection
remaining_features = list(X.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()


# In[17]:


#Iterating adding next most significant variable to equation
for i in range(1,k+1):
    best_RSS = np.inf
    for combo in itertools.combinations(remaining_features,1):
        
            RSS = fit_forest_reg(X[list(combo) + features],Y)   #Store temp result 
            
            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]
                
 #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()


# In[18]:


#Creating a Dataframe with combined features RSS and R2 values 
df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df1['numb_features'] = df1.index

#Initializing useful variables
m = len(Y)
p = 12
hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])


# In[19]:


#Computing
df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
df1['R_squared_adj'].idxmax()
df1['R_squared_adj'].max()
variables = ['C_p', 'AIC','BIC','R_squared_adj']
fig = plt.figure(figsize = (18,6))


# In[20]:


#Marking most significant value in each graph
for i,v in enumerate(variables):
    ax = fig.add_subplot(1, 4, i+1)
    ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
    ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
    if v == 'R_squared_adj':
        ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
    else:
        ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
    ax.set_xlabel('Number of predictors')
    ax.set_ylabel(v)

fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
plt.show()


# In[21]:


###DEFINING REGRESSION MODELS USED###
#Linear Regression Model in SKLearn
linear = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
lr = linear.fit(x_train, y_train)
predict_LR = lr.predict(x_test)

#Linear Regression Model in Statsmodel for visualisation
x_train_const = sm.add_constant(x_train)
ols = sm.OLS(y_train, x_train_const)
olsreg = ols.fit()
print(olsreg.summary())

#Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
predict_Ridge = ridge.predict(x_test)

#Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(x_train, y_train)
predict_Lasso = lasso.predict(x_test)
  
#Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(x_train, y_train)
predict_Percp = perceptron.predict(x_test)

#Random Forest Regressor
regr = RandomForestRegressor(n_estimators=10,max_depth=4,random_state=0)
regr.fit(x_train, y_train)
cols = regr.feature_importances_
predict_RF = regr.predict(x_test)

#Producing a Dataframe with R2 value for each regression model
results = pd.DataFrame({
    'Model': ['Linear Regression','Ridge Regression','Lasso Regression',
              'Perceptron', 'Random Forest'],
    'Score': [lr.score(x_train, y_train), ridge.score(x_train, y_train), lasso.score(x_train, y_train),
              perceptron.score(x_train, y_train), regr.score(x_train, y_train)
              ]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(8)

#Defining the importance of each variable to the Random Forest Regression
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(regr.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()




# In[22]:


############################################PARAMETER OPTIMISATION##########################################


# In[23]:


#Random Forest Regressor Parameter Optimisation
rf = RandomForestRegressor(n_estimators=10,
                           min_samples_split=5,
                           min_samples_leaf=4,
                           max_features='auto',
                           max_depth=4,
                           bootstrap=True,
                           oob_score=True,
                           random_state=0)
rf.fit(x_train, y_train)
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \n'.format(rf.score(x_train, y_train), rf.oob_score_))
rf_predict = rf.predict(x_test)


# In[24]:



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, cv = 3, n_jobs=-1)

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(y_train, x_train)
best_params = rf_random.best_params_


# In[ ]:





# In[ ]:





# In[ ]:




