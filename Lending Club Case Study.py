#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing all relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#reading the data
loan_data = pd.read_csv("loan.csv")
loan_data.head()


# In[12]:


loan_data.shape


# In[5]:


loan_data.isnull().sum()


# # As we can see Clearly there is high number of null values
# Lets Drop them, first
# 

# In[13]:


loan_data.dropna(axis = 1, how = 'all', inplace = True)
loan_data.head()


# # There are several columns which are single valued.
# They cannot contribute to our analysis in any way. So removing them.

# In[14]:


loan_data.drop(['pymnt_plan', "initial_list_status",'collections_12_mths_ex_med','policy_code','acc_now_delinq', 'application_type', 'pub_rec_bankruptcies', 'tax_liens', 'delinq_amnt'], axis = 1, inplace = True)
loan_data.head()


# In[15]:


columns = loan_data.columns.tolist()
columns


# # Now we have 48 columns out of which some correspond to the post approval of loan
# We are analyzing the user details and the driving factors of loan defaulting before approving loan.
# 
# So we can safely remove the columns / variables corresponding to that scenario.
# 
# Also there are some columns such as "id", "member_id", "url", "title", "emp_title", "zip_code", "last_credit_pull_d", "addr_state".
# 
# The above features or columns doesnt contribute to the loan defaulting in any way due to irrelevant information. So removing them.
# 
# "desc" has description (text data) which we cannot do anythhing about for now. So removing the column.
# 
# "out_prncp_inv" , "total_pymnt_inv " are useful for investors but not contributing to the loan defaulting analysis. So removing them.
# 
# "funded_amnt" is not needed because we only need info as to how much is funded in actual. As we have "funded_amnt_inv" , we can remove the earlier column.
# 
# 

# # List of post-approval features
# 
# delinq_2yrs
# 
# revol_bal
# 
# out_prncp
# 
# total_pymnt
# 
# total_rec_prncp
# 
# total_rec_int
# 
# total_rec_late_fee
# 
# recoveries
# 
# collection_recovery_fee
# 
# last_pymnt_d
# 
# last_pymnt_amnt
# 
# next_pymnt_d
# 
# chargeoff_within_12_mths
# 
# mths_since_last_delinq
# 
# mths_since_last_record

# In[17]:


loan_data.columns


# In[22]:


# Dropping the specified columns from the loan_data DataFrame
loan_data.drop(["id", "member_id", "url", "title", "emp_title", "zip_code", "last_credit_pull_d", "addr_state", 
                "desc", "out_prncp_inv", "total_pymnt_inv", "funded_amnt", "delinq_2yrs", "revol_bal", 
                "out_prncp", "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", 
                "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", 
                "chargeoff_within_12_mths", "mths_since_last_delinq", "mths_since_last_record"], axis=1, inplace=True,  errors='ignore')

# Display the shape of the modified DataFrame
print(loan_data.shape)

# Display the first few rows of the modified DataFrame
print(loan_data.head())


# # The goal of the analysis is to see who is likely to default and this can only be said in case of either fully paid or charged off loans.
# We cannot make up anything for current loans
# 
# To exclude that data , removing the records with current loan status

# In[23]:


loan_data = loan_data[loan_data.loan_status != "Current"]
loan_data.loan_status.unique()


# In[24]:


#checking for missing vales
(loan_data.isna().sum()/len(loan_data.index))*100


# In[25]:


#handling them by checking info first
loan_data.info()


# In[26]:


print("Mode : " + loan_data.emp_length.mode()[0])
loan_data.emp_length.value_counts()


# # The above value counts shows that the mode value has far higher frequency than that of the next most frequent value
# This means that we can safely assign the value of mode to the null values in the column.
# 
# Also the missing values are in very low percentage. So imputung with mode value doesnt affect the analysis much

# In[28]:


loan_data.emp_length.fillna(loan_data.emp_length.mode()[0], inplace = True)
loan_data.emp_length.isna().sum()


# In[29]:


loan_data.dropna(axis = 0, subset = ['revol_util'] , inplace = True)
loan_data.revol_util.isna().sum()


# # Standardizing the data
# "revol_util" column although described as an object column, it has continous values.
# 
# So we need to standardize the data in this column
# 
# "int_rate" is one such column.
# 
# "emp_length" --> { (< 1 year) is assumed as 0 and 10+ years is assumed as 10 }
# 
# Although the datatype of "term" is arguable to be an integer, there are only two values in the whole column and it might as well be declared a categorical variable.

# In[30]:


loan_data.revol_util = pd.to_numeric(loan_data.revol_util.apply(lambda x : x.split('%')[0]))


# In[31]:


loan_data.int_rate = pd.to_numeric(loan_data.int_rate.apply(lambda x : x.split('%')[0]))


# In[35]:


loan_data.emp_length = pd.to_numeric(loan_data.emp_length.apply(lambda x: 0 if "<" in x else (x.split('+')[0] if "+" in x else x.split()[0])))


# In[34]:


loan_data.head()


# # Outlier Treatment

# In[37]:


sns.boxplot(loan_data['annual_inc'])


# # Clearly indincating the presence of outliers.
# So, Removing them.
# Let's see the quantile info and take an appropriate action.
# The values after 95 percentile seems to be disconected from the general distribution and also there is huge increase in the value for small quantile variation.
# So, considering threshold for removing outliers as 0.95
# 

# In[38]:


quantile_info = loan_data.annual_inc.quantile([0.5, 0.75,0.90, 0.95, 0.97,0.98, 0.99])
quantile_info


# In[39]:


per_95_annual_inc = loan_data['annual_inc'].quantile(0.95)
loan_data = loan_data[loan_data.annual_inc <= per_95_annual_inc]


# In[40]:


sns.boxplot(loan_data.annual_inc)


# # Now the "annual_inc" data looks good and proceeding next.
# Let's analyze other numerical variables which could possibly have outliers.
# dti
# loan_amnt
# funded_amnt_inv

# In[41]:


sns.boxplot(loan_data.dti)


# In[42]:


sns.boxplot(loan_data.loan_amnt)


# In[43]:


loan_data.loan_amnt.quantile([0.75,0.90,0.95,0.97,0.975, 0.98, 0.99, 1.0])


# In[44]:


sns.boxplot(loan_data.funded_amnt_inv)


# In[45]:


loan_data.funded_amnt_inv.quantile([0.5,0.75,0.90,0.95,0.97,0.975, 0.98,0.985, 0.99, 1.0])


# Though there are some values far from distribution, the distribution is pretty continousand there is no need to remove outliers / extreme values for these above columns.

# # Visualizing Categorical Data
# ##As we already have grade column, extracting only subgrade (int level value) from the sub_grade variable
# We are analyzing and visualizing only the defaulter data. So subsetting the data while plotting only for 'Charged Off' loan_status for below plots

# In[47]:


sns.countplot(x = 'loan_status', data = loan_data)


# In[48]:


loan_data.sub_grade = pd.to_numeric(loan_data.sub_grade.apply(lambda x : x[-1]))
loan_data.sub_grade.head()


# In[49]:


fig, ax = plt.subplots(figsize=(12,7))
sns.set_palette('colorblind')
sns.countplot(x = 'grade', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] , hue = 'sub_grade',data = loan_data[loan_data.loan_status == 'Charged Off'])


# In[50]:


sns.countplot(x = 'grade', data = loan_data[loan_data.loan_status == 'Charged Off'], order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# # Analyzing home_ownership

# In[51]:


#checking unique values for home_ownership
loan_data['home_ownership'].unique()


# ### There are only 3 records with 'NONE' value in the data. So replacing the value with 'OTHER'

# In[52]:


#replacing 'NONE' with 'OTHERS'
loan_data['home_ownership'].replace(to_replace = ['NONE'],value='OTHER',inplace = True)


# In[53]:


#checking unique values for home_ownership again
loan_data['home_ownership'].unique()


# In[54]:


fig, ax = plt.subplots(figsize = (6,4))
ax.set(yscale = 'log')
sns.countplot(x='home_ownership', data=loan_data[loan_data['loan_status']=='Charged Off'])


# # Analyzing purpose

# In[91]:


fig, ax = plt.subplots(figsize = (8,6))
ax.set(xscale = 'log')
sns.countplot(y ='purpose', data=loan_data[loan_data.loan_status == 'Charged Off'])


# # Creating bins for some numerical variable to make them categorical

# In[56]:


#creating bins for int_rate,open_acc,revol_util,total_acc
loan_data['int_rate_groups'] = pd.cut(loan_data['int_rate'], bins=5,precision =0,labels=['5%-9%','9%-13%','13%-17%','17%-21%','21%-24%'])
loan_data['open_acc_groups'] = pd.cut(loan_data['open_acc'],bins = 5,precision =0,labels=['2-10','10-19','19-27','27-36','36-44'])
loan_data['revol_util_groups'] = pd.cut(loan_data['revol_util'], bins=5,precision =0,labels=['0-20','20-40','40-60','60-80','80-100'])
loan_data['total_acc_groups'] = pd.cut(loan_data['total_acc'], bins=5,precision =0,labels=['2-20','20-37','37-55','55-74','74-90'])
loan_data['annual_inc_groups'] = pd.cut(loan_data['annual_inc'], bins=5,precision =0,labels =['3k-31k','31k-58k','58k-85k','85k-112k','112k-140k'])


# In[57]:


# Viewing new bins created
loan_data.head()


# ### Analyzing interest rate wrt the interest rate bins created

# In[58]:


fig, ax = plt.subplots(figsize = (15,10))
plt.subplot(221)
sns.countplot(x='int_rate_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])
plt.xlabel('Interest Rate')
plt.subplot(222)
sns.countplot(x='emp_length', data=loan_data[loan_data.loan_status == 'Charged Off'])


# ### Similarly analyzing open_acc,revol_util,total_acc,annual_inc

# In[59]:


fig, ax = plt.subplots(figsize = (7,5))
ax.set_yscale('log')
sns.countplot(x='open_acc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[60]:


sns.countplot(x='revol_util_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[61]:


fig, ax = plt.subplots(figsize = (8,6))
ax.set_yscale('log')
sns.countplot(x='total_acc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[62]:


fig, ax = plt.subplots(figsize = (10,6))
sns.countplot(x='annual_inc_groups', data=loan_data[loan_data.loan_status == 'Charged Off'])


# In[63]:


sns.countplot(y='term', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[64]:


sns.countplot(x='verification_status', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[65]:


fig,ax = plt.subplots(figsize = (10,8))
ax.set_yscale('log')
sns.countplot(x='inq_last_6mths', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[66]:


fig,ax = plt.subplots(figsize = (7,5))
ax.set_yscale('log')
sns.countplot(x='pub_rec', data=loan_data[loan_data['loan_status']=='Charged Off'])


# ## Analyzing by issued month and year

# In[67]:


## Extracting month and year
df_month_year = loan_data['issue_d'].str.partition("-", True)     
loan_data['issue_month']=df_month_year[0]                       
loan_data['issue_year']='20' + df_month_year[2]


# In[68]:


loan_data.head()


# In[69]:


plt.figure(figsize=(15,15))
plt.subplot(221)
sns.countplot(x='issue_month', data=loan_data[loan_data['loan_status']=='Charged Off'])
plt.subplot(222)
sns.countplot(x='issue_year', data=loan_data[loan_data['loan_status']=='Charged Off'])


# # Maximum number of defaults occured when the loan was sanctioned/issued in Dec. Loan issued in the year 2011 were also as compared to other years

# ### Analyzing installment,dti, loan_amnt

# In[70]:


loan_data['installment_groups'] = pd.cut(loan_data['installment'], bins=10,precision =0,labels=['14-145','145-274','274-403','403-531','531-660','660-789','789-918','918-1047','1047-1176','1176-1305'])
loan_data['funded_amnt_inv_group'] = pd.cut(loan_data['funded_amnt_inv'], bins=7,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k']) ## bin is starting from -35?
loan_data['loan_amnt_groups'] = pd.cut(loan_data['loan_amnt'], bins=7,precision =0,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k'])
loan_data['dti_groups'] = pd.cut(loan_data['dti'], bins=5,precision =0,labels=['0-6','6-12','12-18','18-24','24-30'])


# In[71]:


fig,ax = plt.subplots(figsize = (12,5))
ax.set_yscale('log')
sns.countplot(x='funded_amnt_inv_group', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[72]:


fig,ax = plt.subplots(figsize = (15,6))
ax.set_yscale('log')
sns.countplot(x='loan_amnt_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[73]:


sns.countplot(x='dti_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# In[74]:


fig,ax = plt.subplots(figsize = (15,6))
ax.set_yscale('log')
sns.countplot(x='installment_groups', data=loan_data[loan_data['loan_status']=='Charged Off'])


# # Observations

# ## The above analysis with respect to the charged off loans for each variable suggests the following. There is a more probability of defaulting when :
# 
# 
Applicants having house_ownership as 'RENT'
Applicants who use the loan to clear other debts
Applicants who receive interest at the rate of 13-17%
Applicants who have an income of range 31201 - 58402
Applicants who have 20-37 open_acc
Applicants with employement length of 10
When funded amount by investor is between 5000-10000
Loan amount is between 5429 - 10357
Dti is between 12-18
When monthly installments are between 145-274
Term of 36 months
When the loan status is Not verified
When the no of enquiries in last 6 months is 0
When the number of derogatory public records is 0
When the purpose is 'debt_consolidation'
Grade is 'B'
And a total grade of 'B5' level.
# ### Also there is a very interesting observation from the date issued. The late months of an year indicated the high possibility of defaulting.
The high number of loan defaults in 2011 could be due to the financial crisis in USA (Assuming the data is of US origin)
# # Analysing annual income with other columns for more insights

# ### 1.Annual income vs loan purpose

# In[77]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='annual_inc', y='purpose', hue ='loan_status',palette="deep")
plt.show()


# ### Though the number of loans applied and defaulted are the highest in number for "debt_consolation", the annual income of those who applied isn't the highest.
Applicants with higher salary mostly applied loans for "home_improvment", "house", "renewable_energy" and "small_businesses"
# ### 2.Annual income vs home ownership

# In[78]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='home_ownership', y='annual_inc', hue ='loan_status',palette="pastel")
plt.show()


# ### Annual Income vs Loan amount

# In[79]:


sns.barplot(x = "annual_inc_groups", y = "loan_amnt", hue = 'loan_status', data = loan_data)


# ## Across all the income groups, the loan_amount is higher for people who defaulted.

# In[80]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='int_rate_groups', y='annual_inc', hue ='loan_status',palette="pastel")
plt.show()


# ## Analysing loan_amount with other columns for more insights

# ### 1.Loan Amount vs Interest Rate

# In[81]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt_groups', y='int_rate', hue ='loan_status',palette="pastel")
plt.show()


# ### 2.Loan vs Loan purpose

# In[82]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt', y='purpose', hue ='loan_status',palette="pastel")
plt.show()


# ### 3.Loan vs House Ownership

# In[83]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt', y='home_ownership', hue ='loan_status',palette="pastel")
plt.show()


# ### 4.Loan amount vs month issued and year issued

# In[84]:


plt.figure(figsize=(20,20))
plt.subplot(221)
sns.lineplot(data =loan_data,y='loan_amnt', x='issue_month', hue ='loan_status',palette="pastel")
plt.subplot(222)
sns.lineplot(data =loan_data,y='loan_amnt', x='issue_year', hue ='loan_status',palette="pastel")


# ### 5.Loan amount vs Grade

# In[85]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt', y='grade', hue ='loan_status',palette="pastel", order=['A','B','C','D','E','F','G'])
plt.show()


# In[86]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='loan_amnt', y='grade', hue ='loan_status',palette="pastel", order=['A','B','C','D','E','F','G'])
plt.show()


# In[87]:


plt.figure(figsize=(20,20))
plt.subplot(221)
sns.barplot(data =loan_data,y='loan_amnt', x='emp_length', hue ='loan_status',palette="pastel")
plt.subplot(222)
sns.barplot(data =loan_data,y='loan_amnt', x='verification_status', hue ='loan_status',palette="pastel")


# ## Employees with longer working history got the loan approved for a higher amount.

# ##### Looking at the verification status data, verified loan applications tend to have higher loan amount. Which might indicate that the firms are first verifying the loans with higher values.

# #### Grade vs interest rate

# In[88]:


plt.figure(figsize=(10,10))
sns.barplot(data =loan_data,x='int_rate', y='grade', hue ='loan_status',palette="pastel", order=['A','B','C','D','E','F','G'])
plt.show()


# In[89]:


# fig,ax = plt.subplots(figsize = (15,6))
plt.tight_layout()
sns.catplot(data =loan_data,y ='int_rate', x ='loan_amnt_groups', hue ='loan_status',palette="pastel",kind = 'box')


# ## The interest rate for charged off loans is pretty high than that of fully paid loans in all the loan_amount groups

# ### This can be a pretty strong driving factor for loan defaulting.

# In[90]:


sns.catplot(x = 'term', y = 'loan_amnt', data = loan_data,hue = 'loan_status', kind = 'bar')


# ## Applicants who applied and defaulted have no significant difference in loan_amounts.

# ### Which means that applicants applying for long term has applied for more loan.

# # Observations

# ## The above analysis with respect to the charged off loans. There is a more probability of defaulting when :

# ### Applicants taking loan for 'home improvement' and have income of 60k -70k
# 
# ### Applicants whose home ownership is 'MORTGAGE and have income of 60-70k
# 
# ### Applicants who receive interest at the rate of 21-24% and have an income of 70k-80k
# 
# ### Applicants who have taken a loan in the range 30k - 35k and are charged interest rate of 15-17.5 %
# 
# ### Applicants who have taken a loan for small business and the loan amount is greater than 14k
# 
# ### Applicants whose home ownership is 'MORTGAGE and have loan of 14-16k
# 
# ### When grade is F and loan amount is between 15k-20k
# 
# ### When employment length is 10yrs and loan amount is 12k-14k
# 
# ### When the loan is verified and loan amount is above 16k
# 
# ### For grade G and interest rate above 20%

# In[ ]:




