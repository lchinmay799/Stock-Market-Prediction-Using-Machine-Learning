#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas_datareader as pdr


# In[5]:


df=pdr.get_data_tiingo('GOOG',api_key='600965445e480ded65188f8485444e6643e71e2a')


# In[6]:


df.to_csv('GOOG.csv')


# In[7]:


import pandas as pd


# In[9]:


df=pd.read_csv('GOOG.csv')


# In[10]:


df


# In[11]:


df1=df.reset_index()['close']


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


plt.plot(df1)


# In[14]:


df1.shape


# In[15]:


import numpy as np


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[17]:


df1


# In[18]:


train_size=int(len(df1)*0.65)
test_size=len(df1)-train_size
train_set,test_set=df1[0:train_size,:],df1[train_size:,:]


# In[19]:


train_set.shape,test_set.shape


# In[20]:


import numpy as np
def get_charset(dataset,time_step=1):
    datax,datay=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return np.array(datax),np.array(datay)


# In[21]:


time_step=100
x_train,y_train=get_charset(train_set,time_step)
x_test,y_test=get_charset(test_set,time_step)


# In[22]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[23]:


x_test.shape


# In[24]:


x_train.shape


# In[25]:


from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[26]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[27]:


model.summary()


# In[28]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[29]:


import tensorflow as tf


# In[30]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[31]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[32]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(train_predict,y_train))


# In[33]:


math.sqrt(mean_squared_error(test_predict,y_test))


# In[34]:


train_predict.size


# In[35]:


import numpy as np
look_back=100
trainPredictPlot=np.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=np.empty_like(df1)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)


# In[36]:


len(test_set)


# In[45]:


x_input=test_set[341:].reshape(1,-1)


# In[46]:


x_input.shape


# In[47]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[48]:


len(temp_input)


# In[49]:


import numpy as np
n_steps=100
i=0
list_output=[]
while i<30:
    if len(temp_input)>100:
        x_input=np.array(temp_input[1:])
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape(1,n_steps,1)
        print("{} day input {}".format(i,x_input))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        list_output.extend(yhat.tolist())
        temp_input=temp_input[1:]
        i+=1
    else:
        x_input=x_input.reshape(1,n_steps,1)
        yhat=model.predict(x_input)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i+=1
print(list_output)


# In[50]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[52]:


import matplotlib.pyplot as plt
plt.plot(day_new,scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred,scaler.inverse_transform(list_output))


# In[53]:


df3=df1.tolist()
df3.extend(list_output)


# In[54]:


plt.plot(df3[1:])


# In[ ]:




