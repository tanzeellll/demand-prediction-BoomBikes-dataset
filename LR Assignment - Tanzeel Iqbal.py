#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings ('ignore')


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


ds = pd.read_csv('day.csv', skipinitialspace=True)


# In[34]:


ds.head()


# In[4]:


ds.shape


# In[5]:


ds.info()


# In[6]:


ds.describe()


# In[11]:


ds.nunique().sort_values()


# # PS 
# 
# ### Categorical variables are yr, holiday, workingday, weathersit, season, weekday, mnth
# 
# ### Continuous variable are temp, hum, casual, windspeed, registered, atemp, cnt, instant, dteday        

# In[36]:


#Changing variables from 0s and 1s to its actual representation for EDA purposes. We will be changing season, mnth, weathersit, weekday

ds['season']=ds.season.map({1: 'spring', 2: 'summer',3:'fall', 4:'winter'})
ds['mnth'] = ds.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
ds['weathersit'] = ds.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow',4:'Snow & Fog'})
ds['weekday'] = ds.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})


# In[37]:


ds.head()


# In[42]:


#plotting for continuous var

sns.pairplot(ds, vars=["temp", "hum",'casual','windspeed','registered','atemp','cnt','instant'])
plt.show()


# In[ ]:


#By the above plots, we can see correlations between variables, temp for a start has strong correlation with cnt


# In[57]:


#plotting for categorical var


# In[49]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'yr', y = 'cnt', data = ds)

#PS bike rented was more in 2019


# In[50]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'holiday', y = 'cnt', data = ds)

#PS bike rental is significantly more on holidays


# In[51]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'workingday', y = 'cnt', data = ds)

#PS bike rental is not affected by working days


# In[60]:


plt.figure(figsize=(4, 3))
sns.boxplot(x = 'weekday', y = 'cnt', data = ds)

#PS bike rental is more on Mon, Fri and Sun


# In[56]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'mnth', y = 'cnt', data = ds)


# In[52]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'weathersit', y = 'cnt', data = ds)


# In[53]:


plt.figure(figsize=(5, 3))
sns.boxplot(x = 'season', y = 'cnt', data = ds)

#PS bike rental is more diring summer and fall


# In[63]:


#Rental VS Season

sns.barplot(y='cnt', x='season', data=ds)
plt.show()


# In[65]:


#Rental VS Weather

sns.barplot(y='cnt', x='weathersit', data=ds)
plt.show()


# In[68]:


#mm yy vs Count

sns.barplot(y='cnt', x='mnth', hue='yr', data=ds)
plt.show()


# In[72]:


#Heatmap to check for correlation between variables

plt.figure(figsize=(15,15))
sns.heatmap(ds.corr(), cmap="Blues", annot=True)
plt.show()


# In[73]:


#Preparing data for model testing
#We will be dropping not so relevant columns
#We will then be converting categorical data i.e weekday, month, weathersit & season to bollean codes(dummy variables)
#We will be splitting dataset into train 70% and test 30% sets. 
#We will then be normalizing values to bring all values as close to and between 0 and 1 to make sure each variable is at par


# In[76]:


ds.head()


# In[250]:


ds = ds.drop(['instant','dteday','casual', 'registered','atemp'], axis=1)
ds.head()


# In[77]:


ds.head()


# In[79]:


months=pd.get_dummies(ds.mnth,drop_first=True)
weekdays=pd.get_dummies(ds.weekday,drop_first=True)
weather_sit=pd.get_dummies(ds.weathersit,drop_first=True)
seasons=pd.get_dummies(ds.season,drop_first=True)


# In[81]:


ds=pd.concat([months,weekdays,weather_sit,seasons,ds],axis=1)
ds.head()


# In[83]:


ds.drop(['season','mnth','weekday','weathersit'], axis = 1, inplace = True)
ds.head()


# In[84]:


from sklearn.model_selection import train_test_split

ds_train, ds_test = train_test_split(ds, train_size = 0.7, random_state = 100)


# In[85]:


print(ds_train.shape)
print(ds_test.shape)


# In[87]:


from sklearn.preprocessing import MinMaxScaler

#Normalisation = (x-xmin)/(x max-x min)
#Standardisation= (x-mu)/ sigma


# In[89]:


scaler = MinMaxScaler()

s_var = ['temp', 'hum', 'windspeed', 'cnt']

ds_train[s_var] = scaler.fit_transform(ds_train[s_var])


# In[90]:


ds_train.head()


# In[92]:


ds_train.describe()


# In[93]:


plt.figure(figsize=(25, 20))
sns.heatmap(ds_train.corr(),cmap='Reds',annot = True)
plt.show()


# In[95]:


#Building the model


# In[96]:


y_train = ds_train.pop('cnt')
X_train = ds_train


# In[97]:


ds_train.head()


# In[101]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[108]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[113]:


rfe = RFE(lm, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[114]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[116]:


#RFE support true

X_train.columns[rfe.support_]


# In[117]:


#RFE support false

X_train.columns[~rfe.support_]


# In[152]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]


# In[153]:


X_train_rfe.head()


# In[154]:


#Since constant is not added, we have too add it using statsmodels

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)


# In[155]:


lm = sm.OLS(y_train,X_train_rfe).fit()


# In[156]:


print(lm.summary())


# In[157]:


X_train_rfe = X_train_rfe.drop(['const'], axis=1)


# In[158]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[159]:


#from the above calculations, we have decided to drop July due to its low significance and high VIF

X_train_rfe1 = X_train_rfe.drop(["July"], axis=1)


# In[196]:


#rebuilding model again without July

X_train_rfe1 = sm.add_constant(X_train_rfe1)
lm1 = sm.OLS(y_train,X_train_rfe1).fit()
print(lm1.summary())


# In[197]:


X_train_rfe1 = X_train_rfe1.drop(["const"], axis=1)


# In[198]:


vif = pd.DataFrame()
X = X_train_rfe1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[199]:


#from the above calculations, we have decided to drop Jan due to its low significance and high VIF

X_train_rfe2 = X_train_rfe1.drop(["Jan"], axis=1)


# In[237]:


#rebuilding model again without Jan

X_train_rfe2 = sm.add_constant(X_train_rfe2)
lm2 = sm.OLS(y_train,X_train_rfe2).fit()
print(lm2.summary())


# In[168]:


X_train_rfe2 = X_train_rfe2.drop(['const'], axis=1)


# In[169]:


vif = pd.DataFrame()
X = X_train_rfe2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[201]:


#from the above calculations, we have decided to drop hum due to its low significance and high VIF

X_train_rfe3 = X_train_rfe2.drop(["hum"], axis=1)


# In[238]:


#rebuilding model again without hum

X_train_rfe3 = sm.add_constant(X_train_rfe3)
lm3 = sm.OLS(y_train,X_train_rfe3).fit()
print(lm3.summary())


# In[203]:


X_train_rfe3 = X_train_rfe3.drop(['const'], axis=1)


# In[204]:


vif = pd.DataFrame()
X = X_train_rfe3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[208]:


X_train_rfe3.head()


# In[230]:


#from the above calculations, we have decided to drop Nov due to its low significance

X_train_rfe4=X_train_rfe3.drop(["Nov"], axis=1)


# In[231]:


X_train_rfe4


# In[239]:


#rebuilding model again without Nov

X_train_rfe4 = sm.add_constant(X_train_rfe4)
lm4 = sm.OLS(y_train,X_train_rfe4).fit()
print(lm4.summary())


# In[233]:


#from the above calculations, we have decided to drop Dec due to its low significance

X_train_rfe5 = X_train_rfe4.drop(["Dec"], axis=1)


# In[234]:


X_train_rfe5


# In[240]:


#rebuilding model again without Dec

X_train_rfe5 = sm.add_constant(X_train_rfe5)
lm5 = sm.OLS(y_train,X_train_rfe5).fit()
print(lm5.summary())


# In[245]:


X_train_rfe5 = X_train_rfe5.drop(['const'], axis=1)


# In[246]:


vif = pd.DataFrame()
X = X_train_rfe5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[184]:


X_train_rfe5


# In[252]:


#Applying model on TEST SET

num_vars=['temp','hum','windspeed','cnt']

ds[num_vars] = scaler.transform(ds[num_vars])
ds.head()


# In[254]:


y_test = ds.pop('cnt')
X_test = ds
X_test.describe()


# In[255]:


X_train_rfe5.columns


# In[256]:


X_test_new = X_test[X_train_rfe5.columns]
X_test_new1 = sm.add_constant(X_test_new)
X_test_new1.head()


# In[257]:


y_pred = lm5.predict(X_test_new1)


# In[258]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[259]:


Adj_r2=1-(1-0.8115083)*(11-1)/(11-1-1)
print(Adj_r2)


# In[261]:


fig = plt.figure()
plt.figure(figsize=(15,8))
plt.scatter(y_test,y_pred,color='blue')
fig.suptitle('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)  

plt.show()


# In[262]:


#Below is the best fit line

plt.figure(figsize=(15,8))
sns.regplot(x=y_test, y=y_pred, ci=68, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})

plt.title('y_test vs y_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)
plt.show()


# In[263]:


#We should focus more on Summer & Winter season, August, September month, Weekends, Working days as they have good influence on bike rental
#We should focus more on the temperature
#Spring season, misty+cloudy & light snow seem to have -ve co-efficients, so to boost sales we can runlucrative offers to keep demand stedy

