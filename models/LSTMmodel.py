import random
import time
import numpy as np
import pandas as pd
from datetime import datetime
dates = []
def randomDate():
    frmt = '%d-%m-%Y'

    stime = time.mktime(time.strptime('20-01-2018', frmt))
    etime = time.mktime(time.strptime('23-01-2001', frmt))

    ptime = stime + random.random() * (etime - stime)
    dt = datetime.fromtimestamp(time.mktime(time.localtime(ptime)))
    return dt
    
data=pd.read_csv("Keyboard.csv")

data.info
data['date']=0
for index,row in data.iterrows():
    data.loc[index,'date']=randomDate().date()#("20-01-2018", "23-01-2018").date()
    
data['date'] = pd.to_datetime(data['date'])

data=data.drop('Month',axis=1)

data=data.drop('Year',axis=1)

data.columns
data=data[['date','InventoryToFactoryDistance', 'IsHoliday', 'ModeOfDelivery',
       'ProfitToCompany', 'Quantity', 'RootCause', 'TotalWeight']]

df=data.copy()

df=df.set_index('date')


###### VISUALIZATIONS ##########
df.plot(grid=True)

import matplotlib.pyplot as plt
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
df['ProfitToCompany'].plot(linewidth=0.5)



##################################

df1 = pd.get_dummies(df['ModeOfDelivery'])

df=pd.concat([df,df1],axis=1)

df=df.drop('ModeOfDelivery',axis=1)

df=df.drop('Rail',axis=1)


df.columns

df=df[['InventoryToFactoryDistance', 'IsHoliday', 'ProfitToCompany',
       'Quantity', 'TotalWeight', 'Airways', 'Road', 'Water', 'RootCause']]

x= df.iloc[:, 1:9]
y= df.iloc[:,-1:]

x['IsHoliday']=x['IsHoliday'].astype(int) 

y = pd.get_dummies(y['RootCause'])
y=y.drop('Poor Planning',axis=1)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


x=x.set_index('date')


sc = MinMaxScaler()
x_training_set_scaled = sc.fit_transform(x)
x_final=pd.DataFrame(x_training_set_scaled)
x_final['date']=x['date']
#x_training_set_scaled['date']=x['date']
y_training_set_scaled=sc.fit_transform(y)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

X_train = []
y_train = []
 

#Converting into array
X_train = np.array(x_training_set_scaled) 
y_train = np.array(y_training_set_scaled)

#adding the third dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Importing libraties for the LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer 
X_train.shape
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[0],X_train.shape[1], 1)))

# Adding a second LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))

# Adding a third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))

# Adding a fourth LSTM layer
regressor.add(LSTM(units = 50))

# Adding the output layer
regressor.add(Dense(units = 14))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
#Training with 300 epochs
regressor.fit(X_train, y_train, epochs = 300, batch_size = 32)


#Adding datasets to a frame
frames=[X_train,X_test]
#Concating the dataset
total_dataset=pd.concat(frames)

#Taking all values
inputs= total_dataset.values

#Applying feature scaling
inputs = sc.transform(inputs)
X_test = []
X_test = np.array(inputs)
#adding third dimension
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_cause = regressor.predict(X_test)
predicted_cause = sc.inverse_transform(predicted_cause)