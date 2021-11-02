#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/Bhargaw/Downloads/loans_full_schema.csv')


# In[5]:


df


# In[6]:


pd1 = df.dropna(axis=1,how= 'any')


# In[7]:


pd1


# In[8]:


print(df.isnull().sum())


# In[9]:


for col in pd1.columns:
  print(pd1[col].dtype)


# In[10]:


from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

for column in pd1.columns:
  pd1[column]= le.fit_transform(pd1[column]) 


# In[11]:


pd1


# In[12]:


for col in pd1.columns:
  print(pd1[col].dtype)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt 
Corr_mat = pd1.corr()
fig, ax = plt.subplots(figsize=(32,32))
sns.heatmap(data=Corr_mat, annot=True)


# In[14]:


pd2 = pd1
Corr_mat = Corr_mat.abs()
per = Corr_mat.where(np.triu(np.ones(Corr_mat.shape), k=1).astype(np.bool))
drop = [column for column in per.columns if any(per[column] > 0.90)]

# Drop features 
pd2.drop(drop, axis=1, inplace=True)


# In[15]:


pd2


# In[24]:


df.info()


# # Dataset Description
# * The dataset consists of 10,000 observations with 55 variables. Variables are a combination of quantitative and qualitative problems that address multiple problems.

# # Issues with the dataset
# Simple univariate and bivariate analysis is performed on all variables to diagnose the data. There are following main problems with the data.
# 
# * Quantitative variables may contain significant outliers that need to be filtered out, reducing the number of observations available for analysis.
# * Some analyzes are cumbersome because there are many categories for a particular variable.
# * There are quite a few missing values in various variables. Variables such as Annual_income_joint, debt_to_income_joint, and month_since_last_credit_inquiry are important examples.
# * Some variables are qualitative, but not suitable for use as they contain inconsistent categories

# # Visualization 

# In[25]:


ax = df[['sub_grade', 'interest_rate']].boxplot(figsize=(10,6), by='sub_grade',showfliers=False)
ax.set_xlabel("Loan Sub-grade")
ax.set_ylabel("Interest Rate")
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.set_axisbelow(True)
plt.title('')
_ = plt.suptitle("Interest Rate by Loan\'s Grade\n(without outliers)")


# In[26]:


viz_loan_purpose = df['loan_purpose'].value_counts().to_frame().reset_index()
viz_loan_purpose.columns = ['Loan Purpose', 'Count']
plt.subplots(figsize=(18,8))
sns.barplot(y='Count', x='Loan Purpose', data=viz_loan_purpose)
plt.ylabel('Count')
plt.title('Distribution of Loan Purpose')
plt.show()


# In[28]:


Corr_mat = pd1.corr()
fig, ax = plt.subplots(figsize=(32,32))
sns.heatmap(data=Corr_mat, annot=True)


# In[29]:


ax = df[['homeownership', 'annual_income']].boxplot(figsize=(9,6), by='homeownership',showfliers=False)
ax.set_xlabel("Homeownership")
ax.yaxis.grid(True, color='#EEEEEE')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_ticks_position('none')
ax.set_axisbelow(True)
plt.title('')
_ = plt.suptitle('Annual Income by Homeownership\n(without outliers)')


# In[39]:


sns.distplot(df["interest_rate"], rug=True,hist=False);
plt.title("Distribution of Interest Rate");
plt.ylabel("Density");


# # Model

# In[30]:


Y = pd2['interest_rate']


# In[31]:


X = pd2.drop(['interest_rate'], axis=1)


# In[32]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
st_X = scaler.transform(X)


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(st_X, Y, test_size=0.2)


# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
regressor = LinearRegression()  
regressor.fit(x_train, y_train)


# In[35]:


y_pred = regressor.predict(x_test)


# In[36]:


error = (1/(2*(x_test.shape[0]))*np.sum(np.abs(y_pred - y_test)))


# In[37]:


error


# In[56]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / y_test)
# Calculate and display accuracy. i tried to calculate overall how much accurate is my model its not literal accuracy measure
accuracy = 100 - np.mean(mape)


# # Improvements
# * If you have more time, talk to a domain expert to learn more about how a loan is valued based on different characteristics (interest rates should be determined in the same way). 
# * Sepnd more time on studing different features
# * Try to get more amount of data so that I can build the model with more accuracy.
# * I Will try to implement different Models and figure out which model has less error.
# * Also, I can combine multiple model to make one complex model which can fits features more accurately. 

# # CASE STUDY 2

# In[82]:


import pandas as pd
df = pd.read_csv('C:/Users/Bhargaw/Downloads/stout/casestudy.csv')


# In[60]:


df 


# In[69]:


current_year = list(df['year'].unique())
current_year = max(current_year)
current_year
df_2017 = df[df['year'] == current_year]
df_2016 = df[df['year'] == 2016]
df_2017


# In[70]:


# Total revenue for the current year
total_rev_2017 = df_2017['net_revenue'].sum()
total_rev_2017


# In[71]:


# New Customer Revenue e.g. new customers not present in previous year only
filtered_df = df[(df['year'] < current_year)]
    
# selecting all the users who were present in previous years and created the list of users email id
emails_ls = list(filtered_df['customer_email'])
    
# Here selecting only those users who does not present in previous years
inverse_boolean_series = ~(df.customer_email.isin(emails_ls))

filtered_df1 = df[inverse_boolean_series]
    
New_customer_revenue = filtered_df1.net_revenue.sum()
    
print("The revenue of new customer is- {}".format(New_customer_revenue))


# In[72]:


# Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
sum_curr = df_2017['net_revenue'].sum()
sum_prior = df_2016['net_revenue'].sum()

cust_growth = sum_curr - sum_prior
print("The Existing custome growth is-", cust_growth)


# In[73]:


# Existing Customer Revenue Current Year
sum_curr = df_2017['net_revenue'].sum()
print("Existing customer revenue current year-",sum_curr)


# In[74]:


# Existing Customer Revenue Prior Year
sum_prior = df_2016['net_revenue'].sum()
print("Existing customer revenue prior year-",sum_prior)


# In[75]:


# Total Customers Current Year
total_cust_curr = df_2017['customer_email'].count()
print("Total customers current year-",total_cust_curr)


# In[76]:


# Total Customers Previous Year
total_cust_prev = df_2016['customer_email'].count()
print("Total customers previous year-",total_cust_prev)


# In[66]:


filtered_df = df[(df['year'] < current_year)]
    
# selecting all the users who were present in previous years and created the list of users email id
emails_ls = list(filtered_df['customer_email'])
    
# Here selecting only those users who does not present in previous years
inverse_boolean_series = ~(df.customer_email.isin(emails_ls))

filtered_df1 = df[inverse_boolean_series]
    
New_customer_revenue = filtered_df1.net_revenue.sum()
    
print("The revenue of new customer is- {}".format(New_customer_revenue))


# In[77]:


# New Customers
filtered_df = df[(df['year'] < current_year)]
    
# selecting all the users who were present in previous years and created the list of users email id
emails_ls = list(filtered_df['customer_email'])

# Here selecting only those users who does not present in previous years
inverse_boolean_series = ~(df.customer_email.isin(emails_ls))

filtered_df1 = df[inverse_boolean_series]

print("New customers-")
filtered_df1


# In[79]:


rvn_by_year=df.groupby('year')['net_revenue'].sum()
rvn_by_year.name='Total Revenue'
cstm_by_year=df.groupby('year').size()
cstm_by_year.name='Customer'
rvn_cstm_by_year = pd.concat([rvn_by_year,cstm_by_year], axis=1)
ax=rvn_cstm_by_year.plot.bar(secondary_y= 'Customer', rot= 0)
ax.yaxis.grid(True, color='#EEEEEE')
ax.set_axisbelow(True)


# In[80]:


import string
letters = dict(zip(string.ascii_lowercase, [0]*26))
emails=df['customer_email'].unique()
for email in emails:
    for letter in email.split('@')[0]:
        if letter==' ':
            continue
        else:
            letters[letter]+=1
print(letters)


# In[81]:


_=pd.Series(letters).plot(kind='bar', rot=0, title='Occurrence of Each Letter in Email Prefix', figsize=(10,4))

