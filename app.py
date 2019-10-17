#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('../../assets/dataset/pax_20_02_2018_1_CSV.csv')


# In[3]:


df.head()


# In[ ]:


#Stageをpredictするよ！


# In[10]:


df['Stage'].value_counts()


# In[15]:


c


# In[16]:


df.head()


# In[17]:


df['Imp'].value_counts()


# In[18]:


corrs = df.corr()
corrs['Imp'].sort_values()


# In[43]:


log_model = LogisticRegression()


# In[44]:


df.dtypes


# In[52]:


df.dropna(inplace=True)


# In[ ]:





# In[ ]:





# In[53]:


y=df['Imp']
X = df.drop(['Imp', 'Con','Contp','Reg','AgtId','Agt','Dat','Status','Agtp','Stage','StageSub','Part','ThrdPart','OthAgr'], axis=1)


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=4, test_size=.3)


# In[55]:


log_model.fit(X_train, y_train)


# In[56]:


y_preds = log_model.predict(X_test)


# In[57]:


print(y_preds[:5])
print(list(y_test[:5]))


# In[58]:


from sklearn import metrics


# In[59]:


#accuracy
print(metrics.accuracy_score(y_test, y_preds))


# In[60]:


#f1-score
print(metrics.f1_score(y_test,y_preds))


# In[61]:


#confusion matrix
print(metrics.confusion_matrix(y_test,y_preds))


# In[62]:


log_model.coef_


# In[ ]:





# In[ ]:




