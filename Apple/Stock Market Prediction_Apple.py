#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install -c anaconda pandas-datareader


# In[1]:


import pandas_datareader as pdr


# In[2]:


df=pdr.get_data_tiingo('AAPL',api_key='600965445e480ded65188f8485444e6643e71e2a')


# In[3]:


df.to_csv('AAPL.csv')


# In[4]:


import pandas as pd


# In[5]:


df=pd.read_csv('AAPL.csv')


# In[6]:


df.head()


# In[8]:


df.tail()


# In[11]:


df


# In[7]:


df1=df.reset_index()['close']


# df1

# In[8]:


df1


# In[9]:


df1.shape


# In[10]:


import matplotlib.pyplot as plt


# In[16]:


plt.plot(df1)


# In[11]:


import numpy as np


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[13]:


df1.shape


# In[14]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[15]:


training_size,test_size


# In[16]:


train_data


# In[17]:


df


# In[18]:


training_size,test_size


# In[19]:


import numpy
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)
    


# In[20]:


time_step=100
x_train,y_train=create_dataset(train_data,time_step)
x_test,y_test=create_dataset(test_data,time_step)


# In[21]:


print(x_train,y_train,x_test,y_test)


# In[22]:


print(x_train.shape),print(y_train.shape)


# In[23]:


print(x_test.shape),print(y_test.shape)


# In[24]:


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[25]:


print(x_test.shape),print(y_test.shape)


# In[26]:


print(x_train.shape),print(y_train.shape)


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[28]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[29]:


model.summary()


# In[30]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[31]:


import tensorflow as tf


# In[32]:


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[33]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[34]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[35]:


y_test.shape


# In[36]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[ ]:





# In[37]:


look_back=100
trainPredictPlot=numpy.empty_like(df1)
trainPredictPlot[:, :]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :]=train_predict
testPredictPlot=numpy.empty_like(df1)
testPredictPlot[:, :]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show


# In[38]:


len(test_data)


# In[39]:


x_input=test_data[341:].reshape(1,-1)


# In[40]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[41]:


from numpy import array
list_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output{}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        list_output.extend(yhat.tolist())
        i+=1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        list_output.extend(yhat.tolist())
        i+=1
print(list_output)
        
    


# In[42]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


len(df1)


# In[45]:


df3=df1.tolist()
df3.extend(list_output)


# In[47]:


plt.plot(day_new,scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred,scaler.inverse_transform(list_output))


# In[48]:


df3=df1.tolist()
df3.extend(list_output)
plt.plot(df3[1:])


# In[ ]:





# In[ ]:




