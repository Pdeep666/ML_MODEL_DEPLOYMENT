#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


df=pd.read_csv('heart_data.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum().sum()


# In[11]:


df['target'].value_counts()


# In[12]:


# 1--> HEART DISEASE
# 0--> NO HEART DISEASE


# In[13]:


print(pd.crosstab(df['sex'],df['target']))


# In[14]:


correlation_matrix = df.corr()


# In[15]:


plt.figure(figsize=(15, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Sample Heatmap')


# In[16]:


x=df.drop(columns='target',axis=1)
y=df['target']


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1,random_state=1)


# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(x,y)


# In[20]:


# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Training accuracy is:",training_data_accuracy)


# In[21]:


# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on Test data : ', test_data_accuracy)


# In[22]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# In[23]:


import pickle


# In[24]:


pickle.dump(model,open('hmodel.pkl','wb'))


# In[25]:


pickled_model=pickle.load(open('hmodel.pkl','rb'))


# In[ ]:




