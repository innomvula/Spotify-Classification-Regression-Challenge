#!/usr/bin/env python
# coding: utf-8

# # Classification Problem
# 
# #### Create a model for predicting the genre of a song.

# ## Setup

# In[3]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# In[4]:


train = pd.read_csv("Train.csv")


# In[5]:


train.shape


# In[6]:


train.columns


# In[7]:


train.isnull().sum()


# In[8]:


train_c=train.dropna()


# In[9]:


train_c.isnull().sum()


# In[10]:


train_c = train_c.drop(['title', 'artist'],  axis=1)


# In[11]:


train_c['top genre'].value_counts().sort_values().plot.barh()
plt.title('Songs by Top Genre')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.gcf().set_size_inches(10, 30)
plt.show()


# In[12]:


train_c.columns


# In[13]:


corr = train_c.iloc[:, :12].corr()
figure(figsize=(15,15))
ax = sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'hsv')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation Table')
plt.show()


# ## Train/Test Split

# In[14]:


from sklearn import datasets


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


train_c.head(2)


# In[17]:


train_c.columns


# In[18]:


train_c.dtypes


# In[19]:


X, y = train_c[['year', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur',
       'acous', 'spch', 'pop']], train_c['top genre']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


X_train.shape


# In[22]:


X_test.shape


# ## Classifiers

# In[24]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import classification_report as cr


# #### Decision Tree

# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


for a in range(1, 60):
    tree_clf = DecisionTreeClassifier(max_depth=a, criterion="entropy")
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    print(a, accuracy_score(y_test, y_pred))


# In[27]:


tree_clf = DecisionTreeClassifier(max_depth=5, criterion="entropy")
tree_clf.fit(X_train, y_train)
y_pred1 = tree_clf.predict(X_test)
print(tree_clf.__class__.__name__, accuracy_score(y_test, y_pred1))
print(tree_clf.__class__.__name__, cr(y_test, y_pred1))


# #### SVM

# In[28]:


from sklearn.svm import SVC


# In[29]:


svm_clf = SVC(decision_function_shape='ovo', probability=True)
svm_clf.fit(X_train, y_train)
y_pred2 = svm_clf.predict(X_test)
print(svm_clf.__class__.__name__, accuracy_score(y_test, y_pred2))
print(svm_clf.__class__.__name__, cr(y_test, y_pred2))


# #### Random Forest

# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


rnd_clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=200, criterion="gini")
rnd_clf.fit(X_train, y_train)
y_pred3 = rnd_clf.predict(X_test)
print(rnd_clf.__class__.__name__, accuracy_score(y_test, y_pred3))
print(rnd_clf.__class__.__name__, cr(y_test, y_pred3))


# #### Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression


# In[33]:


log_clf = LogisticRegression(penalty = "l2", class_weight = None)
log_clf.fit(X_train, y_train)
y_pred4 = log_clf.predict(X_test)
print(log_clf.__class__.__name__, accuracy_score(y_test, y_pred4))
print(log_clf.__class__.__name__, cr(y_test, y_pred4))


# #### Multi-Layer Perceptron

# In[34]:


from sklearn.neural_network import MLPClassifier


# In[35]:


mlp_clf = MLPClassifier(activation = "tanh", learning_rate = "adaptive")
mlp_clf.fit(X_train, y_train)
y_pred5 = mlp_clf.predict(X_test)
print(mlp_clf.__class__.__name__, accuracy_score(y_test, y_pred5))
print(mlp_clf.__class__.__name__, cr(y_test, y_pred5))


# #### K-Nearest-Neighbours

# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


knn_clf = KNeighborsClassifier(n_neighbors = 11, algorithm = "kd_tree")
knn_clf.fit(X_train, y_train)
y_pred6 = knn_clf.predict(X_test)
print(knn_clf.__class__.__name__, accuracy_score(y_test, y_pred6))
print(knn_clf.__class__.__name__, cr(y_test, y_pred6))


# #### Naive Bayes

# In[38]:


from sklearn.naive_bayes import GaussianNB


# In[39]:


nvb_clf = GaussianNB()
nvb_clf.fit(X_train, y_train)
y_pred7 = nvb_clf.predict(X_test)
print(nvb_clf.__class__.__name__, accuracy_score(y_test, y_pred7))
print(nvb_clf.__class__.__name__, cr(y_test, y_pred7))


# #### Ensemble

# In[40]:


from sklearn.ensemble import VotingClassifier


# In[41]:


voting_clf = VotingClassifier(estimators=[("tree", tree_clf), ("supp", svm_clf), ("forest", rnd_clf), ("log", log_clf), ("percept", mlp_clf), ("neighbour", knn_clf), ("bayes", nvb_clf)],voting='hard')
voting_clf.fit(X_train, y_train)
y_pred8 = voting_clf.predict(X_test)
print(voting_clf.__class__.__name__, accuracy_score(y_test, y_pred8))
print(voting_clf.__class__.__name__, cr(y_test, y_pred8))


# In[42]:


voting2_clf = VotingClassifier(estimators=[("tree", tree_clf), ("supp", svm_clf), ("forest", rnd_clf), ("log", log_clf), ("percept", mlp_clf), ("neighbour", knn_clf), ("bayes", nvb_clf)]
                              ,voting='soft')
voting2_clf.fit(X_train, y_train)
y_pred9 = voting2_clf.predict(X_test)
print(voting2_clf.__class__.__name__, accuracy_score(y_test, y_pred9))
print(voting2_clf.__class__.__name__, cr(y_test, y_pred9))


# ## Comparison Analysis

# In[47]:


scores = []
for a in [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7, y_pred8, y_pred9]:
    scores.append([accuracy_score(y_test, a), precision_score(y_test, a, average = "macro"), recall_score(y_test, a, average = "macro")])


# In[51]:


df = pd.DataFrame(scores, columns = ["Accuracy", "Precision", "Recall"])
df.index = ["Decision Tree", "Support Vector", "Random Forest", "Logistic Regression", 
            "Perceptron", "K-Nearest-Neighbours", "Naïve Bayes", "Hard Voting", "Soft Voting"]
df


# ## Export Solutions

# In[163]:


test = pd.read_csv("Test.csv")


# In[164]:


test.shape


# In[165]:


test.isnull().sum()


# In[166]:


test = test.drop(['title', 'artist'],  axis=1)


# In[287]:


for a in [voting_clf2]:
    predictions = a.predict(test[['year', 'bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur',
       'acous', 'spch', 'pop']])
    solution = test
    solution["top genre"] = predictions
    solution = solution[["Id", "top genre"]]
    print(solution.head())
    solution.to_csv("SOFT_VOTING", index=False)

