#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# <b>Import dataset</b>

# In[4]:


missing_value=["Undefined"]
data_path = "hotel_bookings.csv"
df_hotel = pd.read_csv(data_path, na_values=missing_value)


# In[5]:


df_hotel.head()


# In[6]:


df_hotel.shape


# In[7]:


df_hotel.info()


# <b>Creating a new column by combining the year, month and date of arrival together.</b>

# In[8]:


df_hotel['arrival_date'] = pd.to_datetime(df_hotel.arrival_date_year.astype(str) + '/' + df_hotel.arrival_date_month.astype(str) + '/' + df_hotel.arrival_date_day_of_month.astype(str))


# In[9]:


df_hotel['arrival_date']


# In[10]:


df_hotel['arrival_date'][0]


# <b>Checking how many missing values each column contains</b>

# In[11]:


np.sum(df_hotel.isnull())


# <b>Drop columns we don't need</b>

# In[12]:


df_hotel.drop(columns=["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month", "company", "meal"],
           inplace=True)


# In[14]:


df_hotel.head()


# <b>Filling missing values in agent column</b>

# In[15]:


nan_replacement_dict = {"children": 0 ,"country" : "UKNWN", 'agent' : 0.0, 'company' : 0}
df_hotel.fillna(nan_replacement_dict, inplace = True)


# In[16]:


df_hotel.isnull().sum()


# <b>Data grouping and aggregation</b>

# <b>Average stays during week and on weekends</b>

# In[18]:


df_hotel.stays_in_weekend_nights.mean()


# In[19]:


df_hotel.stays_in_week_nights.mean()


# <b>Group by hotel</b>

# In[20]:


df_hotel.groupby('hotel').stays_in_week_nights.mean()


# <b>Converting datetime columns</b>

# In[21]:


df_hotel['reservation_status_date'] = df_hotel['reservation_status_date'].astype('datetime64')


# In[22]:


df_hotel['arrival_date'] = df_hotel['arrival_date'].astype('datetime64')


# <b>Categorical columns</b>

# In[23]:


categoricals = [i for i in df_hotel.columns if df_hotel.dtypes[i] == 'object']
print("Categorical Columns are: ", *categoricals, sep = '\n')


# In[24]:


for i in categoricals:
    print(("{} : {} Total nunique = {} \n").format(i, df_hotel[i].unique(), df_hotel[i].nunique()))


# <b>Numerical columnns</b>

# In[25]:


numericals = [i for i in df_hotel.columns if df_hotel.dtypes[i] != 'object']
print("Numerical Columns are: ", *numericals, sep = '\n')


# In[26]:


sns.color_palette("Set2", 6)


# In[32]:


sns.set_palette('Set2')
plt.figure(figsize = (8,6))
sns.countplot(x = 'hotel', data = df_hotel, hue = 'stays_in_weekend_nights')
plt.title("No. of stays in weekend nights according to reservation status in both of the hotels")
plt.show()


# There are more stays in weekend nights at the hotels than cancelled reservations. There are very few no-shows.

# In[36]:


_, ax = plt.subplots( nrows = 2, ncols = 1, figsize = (12,10))
sns.countplot(x = 'reservation_status', data = df_hotel, hue = 'stays_in_weekend_nights', ax = ax[0])
sns.countplot(x ='reservation_status', data = df_hotel, hue = 'is_canceled', ax = ax[1])
plt.show()


# City Hotel has more cancellations than Resort Hotel.

# <b>Preprocessing</b>

# In[38]:


df_hotel.drop(['reservation_status', 
'reservation_status_date' , 'arrival_date_week_number',  'arrival_date', 'agent'], axis = 1, inplace = True)
df_hotel.shape


# In[39]:


#Copy of a dataframe
df_hotel_1 = df_hotel.copy()


# In[40]:


hotel = {'Resort Hotel': 0, 'City Hotel' : 1}


# In[41]:


df_hotel_1['hotel'] = df_hotel_1['hotel'].map(hotel)


# In[42]:


df_hotel_1 = pd.get_dummies(data = df_hotel_1, columns = [ 'market_segment', 'distribution_channel','reserved_room_type', 'assigned_room_type', 'customer_type', 'deposit_type'], drop_first = True)


# In[43]:


LE = LabelEncoder()


# In[44]:


df_hotel_1['country'] = LE.fit_transform(df_hotel_1['country'])


# In[45]:


df_hotel_1.head()


# <b>Separate training and target datasets</b>

# In[46]:


X = df_hotel_1.drop('is_canceled', axis = 1)
y = df_hotel_1['is_canceled']


# In[47]:


X_train,  X_test,y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 40)


# <b>Set global random state</b>

# In[48]:


global random_state
random_state = 40


# <b>Dictionary of classification models</b>

# In[50]:


model_dict = {
    'LOR_Model' : LogisticRegression(n_jobs = -1),
    'KNN_Model' : KNeighborsClassifier(),
    'RFC_Model' : RandomForestClassifier(n_jobs = -1),
}  


# In[51]:


def model_1(algorithm, X_train, X_test, y_train, y_test):
    alg = algorithm
    alg_model = alg.fit(X_train, y_train)
    global y_pred
    y_pred = alg_model.predict(X_test)
    
    print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))


# In[52]:


for name, model in model_dict.items():
    print("\n")
    print(name, "\n")
    model_1(model, X_train, X_test, y_train, y_test )


# <b>Conclusion:</b>

# Random Forest Classifier is the best algorithm to predict the cancellation.
# It provides 88% accuracy.
