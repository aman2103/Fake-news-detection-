#!/usr/bin/env python
# coding: utf-8

# # installing  libraries

# In[2]:


pip install Seaborn


# In[3]:


pip install nltk


# # importing libraries 

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import nltk 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,plot_roc_curve


# In[5]:


ps=nltk.PorterStemmer()


# # importing and visualizing data

# In[6]:


data=pd.read_csv("news.csv")


# In[7]:


data.head()


# In[8]:


data.shape


# # splinting   and ploting  data

# In[9]:


x=data['text']
y=data['label']


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[11]:


data["label"].value_counts().plot(kind='bar',color=['salmon','blue'])


# In[12]:


data["label"].value_counts()


# # cleaning and vectorizing data
# 

# In[13]:


def clean_txt(txt):
  txt=" ".join([w for w in txt.split() if len(w)>3 ])
  tokens=txt.split()
  txt=[ps.stem(word) for  word in tokens]
  return txt


# In[14]:


tfidf_vect=TfidfVectorizer(max_df = 0.9 ,min_df =2, max_features =1000 ,stop_words="english" , tokenizer=clean_txt)


# In[15]:


tfidt_fit=tfidf_vect.fit(x_train)
x_train=tfidt_fit.transform(x_train)
x_test=tfidt_fit.transform(x_test)


# In[16]:


x_train.shape,y_train.shape


# In[17]:


x_train.dtype


# # fitting model

# In[18]:


model=PassiveAggressiveClassifier(max_iter=100)
model.fit(x_train,y_train)


# # prediction and score

# In[19]:


preds=model.predict(x_test)


# In[20]:


score=accuracy_score(y_test,preds)


# In[21]:


print(f"Accuracy score : {round(score*100,2)}%")


# In[22]:


print(classification_report(y_test,preds))


# # confusion matrix and roc curve

# In[23]:


fig,ax=plt.subplots(figsize=(7,7))
ax=sns.heatmap(confusion_matrix(y_test,preds),
              annot=True,
              cbar=False)
plt.xlabel('Test')
plt.ylabel('Prediction')


# In[24]:


plot_roc_curve(model,x_test,y_test)


# In[25]:


print(f'roc_auc_score : {model,x_test,y_test}')


# In[ ]:





# In[ ]:




