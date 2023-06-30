#!/usr/bin/env python
# coding: utf-8

# In[4]:



# Importing the required packages for Data Analysis
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[5]:


# Loading the Data set into the dataframe

df = pd.read_csv("Nashville Housing dataset.csv")
pd.set_option('max_columns', 20)


# In[6]:


report = ProfileReport()
report


# In[7]:


df.info( )


# In[8]:


df.head()

# We notice that 5 Attributes namely "LandUse", 'SaleDate', 'SoldAsVacant', 'Owner', 'OwnerAddress', 'TaxDistrict',  are Non numeric text based coloumns.


# In[9]:


df.tail(n=10)


# In[10]:


df.describe()


# In[11]:


df.describe( ).T


# In[12]:


df.shape
print('No. of rows', df.shape[0] ) # total no of rows
print( 'No. of columns', df.shape[1]) # total no of columns


# In[ ]:


print(df.mean('columns'), 'max_columns=19' )


# In[13]:


df.median( )


# In[14]:


df.iloc[56470]


# In[15]:


print(df.isnull( ).sum( ) )

print('Total' , df.isnull( ).sum( ).sum() )

# Notice there are 50% Missing values in 'LandValue' and 'Building Value'


# In[16]:


df.columns

#Creating  a copy of the original dataframe for analysis

df1= df[['UniqueID ',
    'ParcelID',
    'LandUse',
         #'PropertyAddress', 'SaleDate',
       'SalePrice', 
    #'LegalReference', 
    #'SoldAsVacant', 
         'OwnerName',
       #'OwnerAddress', 'Acreage', 'TaxDistrict', 'LandValue', 'BuildingValue',
       'TotalValue', 'YearBuilt', 
         'Bedrooms', 
         #'FullBath',
         'HalfBath' ]].copy()


# In[17]:


df1.shape


# In[18]:


df.copy().shape

df.copy().dtypes


# In[20]:


df4=df.loc[1:1000 ,  ['UniqueID ' , 'ParcelID', 'YearBuilt', 'SaleDate',  'SalePrice', 'LandValue', 'TotalValue', 'LandUse' , 'BuildingValue'] ]


# In[21]:


print(df4)


# In[22]:


from pandas.plotting import scatter_matrix

x = 0
attributes = []
for x in ['UniqueID ' , 'ParcelID', 'YearBuilt', 'SaleDate',  'SalePrice', 'LandValue', 'TotalValue', 'BuildingValue']:
    attributes.append(x)
scatter_matrix(df4[attributes], figsize=(14, 10))
plt.show( )


# In[ ]:


df.[df.PropertyAddress==GOODLETTSVILLE]


# In[23]:


df2 =df1.copy()


# In[24]:


mean_value= df2.mean()
print(mean_value)


# In[25]:


#Creating a subset df4 and imputing with mean values  

df4=df4.fillna(mean_value)


# In[26]:


# Choosing first 25000 records for seeing the correlations between various attributes. 
df4 = df4.iloc[1:25000]


# In[27]:


df4.mean( )


# In[ ]:





# In[28]:


df.loc[df.duplicated(subset=["ParcelID"] )]


# In[29]:


df4['YearBuilt']  .value_counts( )


# In[ ]:


# Plotting graphs with Matplotlib


# In[ ]:


# plt.bar(ax, x_label ='Year Built' , color='red')
plt.plot(ax)
plt.show()


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure(figsize =(9, 9))
X =df4['YearBuilt'].head()
Y =  df4['SalePrice'].head()
plt.bar( X,Y, width=0.4, color= 'green')

plt.axis([0, 10, 0, 12])
plt.title("Year Built v/s Sale Price")
plt.xlabel(" Year Built")
plt.ylabel("Sale Price in '000 $")
plt.show( )


# In[31]:


df4.isnull().sum()


# In[36]:


df4_sort= df4.sort_values(by='LandUse' )


# In[33]:


df4.isnull().sum()


# In[34]:


import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure(figsize =(20, 10))
plt.bar(df4['LandUse'],  df4['SalePrice'], width =0.3,color='maroon')
plt.xlabel(" Sale Price in '000'$")
plt.ylabel("Land Use")
plt.show( )


# In[37]:


x = np.arange(1800,2020,10)
df4.plot(kind='scatter', x= 'LandUse', y= 'SalePrice' , alpha=0.4, cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False, )
plt.xlim(0, 80)
plt.legend(16)
plt.show( )


# In[38]:


df4.max( )


# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure(figsize =(8, 10))
x =df4['YearBuilt']
y =  df4['SalePrice']
df4.plot.scatter( x = df4['YearBuilt'], y=df4['SalePrice'], s=1000, edgecolor= 'green')

plt.axis([0, 10, 0, 20])
plt.title("Year Built v/s Sale Price")
plt.xlabel(" Year Built")
plt.ylabel("Sale Price in '000 $")
plt.show( )


# In[40]:


df4_sort.hist(by='LandUse', figsize=[16, 10], bins=80)
plt.show()


# In[ ]:





# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure(figsize =(16, 10))
ax = plt.gca()
df4.plot(kind='scatter',
        x='LandUse',
        y='LandValue',
        color='green', fontsize=8,ax=ax)
plt.title(' LandUse V/s LandValue')
plt.show( )


# In[43]:


# plt.pie(df4['TotalValue'] , autopct='%1.0f%%')

df4.groupby(['TotalValue']).sum().plot(kind='pie')  
#     autopct='%1.0f%%',  subplots='True')
plt.show( )


# In[44]:


plt.hist(df4['YearBuilt'],edgecolor= 'red')
plt.xlabel('Year Built')
plt.ylabel('No of houses')
plt.show( )


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
fig = plt.figure(figsize =(20, 10))
ax = plt.gca()
df4.plot(kind='scatter',
        x='YearBuilt',
        y='BuildingValue',
        color='lightgreen', fontsize=5,ax=ax)
plt.title(' Year Built V/s BuildingValue')
plt.show( )

