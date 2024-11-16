#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Lets load the Boston House Pricing Dataset

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston=load_boston()


# In[4]:


boston.keys()


# In[5]:


## Lets check the description of the dataset
print(boston.DESCR)


# In[6]:


print(boston.data)


# In[7]:


print(boston.target)


# In[8]:


print(boston.feature_names)


# ## Preparing The Dataset

# In[9]:


dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[10]:


dataset.head()


# In[11]:


dataset['Price']=boston.target


# In[12]:


dataset.head()


# In[13]:


dataset.info()


# In[14]:


## Summarizing The Stats of the data
dataset.describe()


# In[15]:


## Check the missing Values
dataset.isnull().sum()


# In[16]:


### EXploratory Data Analysis
## Correlation
dataset.corr()


# In[17]:


import seaborn as sns
sns.pairplot(dataset)


# ## Analyzing The Correlated Features

# In[18]:


dataset.corr()


# In[19]:


plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")


# In[20]:


plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel("RM")
plt.ylabel("Price")


# In[21]:


import seaborn as sns
sns.regplot(x="RM",y="Price",data=dataset)


# In[22]:


sns.regplot(x="LSTAT",y="Price",data=dataset)


# In[23]:


sns.regplot(x="CHAS",y="Price",data=dataset)


# In[24]:


sns.regplot(x="PTRATIO",y="Price",data=dataset)


# In[25]:


## Independent and Dependent features

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# In[26]:


X.head()


# In[27]:


y


# In[28]:


##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[29]:


X_train


# In[30]:


X_test


# In[31]:


## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[32]:


X_train=scaler.fit_transform(X_train)


# In[33]:


X_test=scaler.transform(X_test)


# In[34]:


import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))


# In[35]:


X_train


# In[36]:


X_test


# ## Model Training

# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


regression=LinearRegression()


# In[39]:


regression.fit(X_train,y_train)


# In[40]:


## print the coefficients and the intercept
print(regression.coef_)


# In[41]:


print(regression.intercept_)


# In[42]:


## on which parameters the model has been trained
regression.get_params()


# In[43]:


### Prediction With Test Data
reg_pred=regression.predict(X_test)


# In[44]:


reg_pred


# ## Assumptions

# In[45]:


## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)


# In[46]:


## Residuals
residuals=y_test-reg_pred


# In[47]:


residuals


# In[48]:


## Plot this residuals 


# In[49]:


## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(reg_pred,residuals)


# In[50]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# ## R square and adjusted R square

# 
# Formula
# 
# **R^2 = 1 - SSR/SST**
# 
# 
# R^2	=	coefficient of determination
# SSR	=	sum of squares of residuals
# SST	=	total sum of squares
# 

# In[51]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[ ]:





# **Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]**
# 
# where:
# 
# R2: The R2 of the model
# n: The number of observations
# k: The number of predictor variables

# In[52]:


#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# ## New Data Prediction

# In[53]:


boston.data[0].reshape(1,-1)


# In[54]:


##transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))


# In[55]:


regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# ## Pickling The Model file For Deployment

# In[56]:


import pickle


# In[57]:


pickle.dump(regression,open('regmodel.pkl','wb'))


# In[58]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[59]:


## Prediction
pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))


# In[ ]:




