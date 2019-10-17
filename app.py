#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
sns.set_style("darkgrid")


# In[106]:


df= pd.read_csv('../../data/pax_20_02_2018_1_CSV.csv')


# In[107]:


df.isnull().sum()


# In[112]:


df= df.drop(['ThrdPart','OthAgr','StageSub','Part'], axis=1)


# In[113]:


df.isnull().sum()


# In[114]:


df['Imp']=df['Stage'].map({'Pre ':0,'SubPar':0,'Imp':1,'Cea':0,'SubComp':0,'Ren':0, 'Oth':0})
df.head()
df.dropna(inplace=True)


# In[115]:


df.shape


# In[116]:


df.dtypes


# In[119]:


X=df.drop(['Con','Contp', 'Reg', 'AgtId', 'Agt','Dat', 'Status', 'Agtp', 'Stage','Imp'], axis=1)


# In[120]:


y = df['Imp'].copy()


# In[121]:


X


# In[123]:


df.dtypes


# In[124]:


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=.1, random_state=42)
X_train.shape


# In[125]:


X.dropna(inplace=True)


# In[126]:


y.value_counts()


# In[127]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)


# In[128]:


y_train.isnull().sum()


# In[129]:


len(y_train)


# In[130]:


y.dropna


# In[131]:


print(len(X.columns))
print(X.columns)
importances = rf.feature_importances_


# In[132]:


#List the features by importance
feat_imp = pd.DataFrame(importances, index=X_test.columns, columns=['importance'])
feat_imp['importance'].sort_values(ascending=False)


# In[133]:


sns.set(style="darkgrid", color_codes=None)
# sns.palplot(sns.color_palette("RdBu", n_colors=7))
ax = top10.plot(kind='bar', legend=False, fontsize=18,  figsize=(15, 7))
plt.xticks(rotation = 45,  fontsize=18)
plt.title('Most Important Predictors',  fontsize=19)
plt.yticks(rotation = 0,  fontsize=18)
plt.ylabel('Feature Importance', rotation=90,  fontsize=18)
plt.savefig('feature_import.png', dpi=300, bbox_inches='tight') 


# In[ ]:




