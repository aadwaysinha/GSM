# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 01:12:16 2019

@author: Aadway
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import time
import numpy as np
from datetime import datetime


def randomDate():
            frmt = '%d-%m-%Y'
        
            stime = time.mktime(time.strptime('20-01-2018', frmt))
            etime = time.mktime(time.strptime('23-01-2001', frmt))
        
            ptime = stime + random.random() * (etime - stime)
            dt = datetime.fromtimestamp(time.mktime(time.localtime(ptime)))
            return dt


def create_model(data):
    
                    
        #data=pd.read_csv("Keyboard.csv")

        data['date']=0
        for index,row in data.iterrows():
            data.loc[index,'date']=randomDate().date()#("20-01-2018", "23-01-2018").date()
            
        data['date'] = pd.to_datetime(data['date'])
        
        data = data.drop('Month',axis=1)
        
        data=data.drop('Year',axis=1)
        
        data.columns
        data=data[['date','InventoryToFactoryDistance', 'IsHoliday', 'ModeOfDelivery',
               'ProfitToCompany', 'Quantity', 'RootCause', 'TotalWeight']]
        
        df=data.copy()
        
        df=df.set_index('date')
        
        
        ###### VISUALIZATIONS ##########
        df.plot(grid=True)
        
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
        
        #Old
        #df=df[['date', 'InventoryToFactoryDistance', 'IsHoliday', 'ProfitToCompany',
        #      'Quantity', 'TotalWeight', 'Airways', 'Road', 'Water', 'RootCause']]
        
        df=df[['InventoryToFactoryDistance', 'IsHoliday', 'ProfitToCompany',
               'Quantity', 'TotalWeight', 'Airways', 'Road', 'Water', 'RootCause']]
        
        
        x= df.iloc[:, 0:8]
        y= df.iloc[:,-1:]
        
        x['IsHoliday']=x['IsHoliday'].astype(int) 
        
        y = pd.get_dummies(y['RootCause'])
        y=y.drop('Poor Planning',axis=1)
        
        
        sc = MinMaxScaler()
        x_training_set_scaled = sc.fit_transform(x)
        #x_final=pd.DataFrame(x_training_set_scaled)
        #x_final['date']=x['date']
        #x_training_set_scaled['date']=x['date']
        y_training_set_scaled=sc.fit_transform(y)
        
        
        #from sklearn.cross_validation import train_test_split
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x_training_set_scaled, y_training_set_scaled, test_size = 1/3, random_state = 0)
        
        X_train = []
        y_train = []
         
        
        #Converting into array
        X_train = np.array(x_training_set_scaled) 
        y_train = np.array(y_training_set_scaled)
        
        #adding the third dimension
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
    #    #Importing libraties for the LSTM model
    #    from keras.models import Sequential
    #    from keras.layers import Dense
    #    from keras.layers import LSTM
    #    from keras.layers import Dropout
        
        # Initialising the RNN
        classifier = Sequential()
        
        # Adding the first LSTM layer 
        #X_train.shape
        classifier.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
        
        # Adding a second LSTM layer 
        classifier.add(LSTM(units = 50, return_sequences = True))
        
        # Adding a third LSTM layer 
        classifier.add(LSTM(units = 50, return_sequences = True))
        
        # Adding a fourth LSTM layer
        classifier.add(LSTM(units = 50))
        
        # Adding the output layer
        classifier.add(Dense(units = 14))
        
        # Compiling the RNN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy')
        
        classifier.fit(X_train, y_train, epochs = 100, batch_size = 32)
        predicted_cause = classifier.predict(X_test)
        predicted_cause = sc.inverse_transform(predicted_cause)
        
        return classifier
    
    
rootCauseDict = {
                     0:"Poor Planning",
                     1:"Strike of workers",
                     2:"Poor Lead time calculation",
                     3:"Poor inventory control",
                     4:"faulty plant layout",
                     5:"excessive machine stoppage",
                     6:"electricty stoppage",
                     7:"Raw material low",
                     8:"material wastage due to over-feeding",
                     9:"Demand Variation",
                     10:"Huge backlog of orders",
                     11:"Supply Shortages and logistical uncertainties",
                     12:"Factory shutdown",
                     13:"Financial problems of company leading to interrupted supplies",
                 }

    
def retMAX(predicted_cause, val):
    #currentRow=''
    df_final=pd.DataFrame(predicted_cause)
    rootcause=''
    maxx = -100000
    for i in range(len(df_final.iloc[val])):
        if df_final.iloc[0][i] > maxx:
            maxx= df_final.iloc[0][i]
            rootcause =rootCauseDict[i]
            
    if(rootcause == ''):
            rootcause = 'Transport Delays'  
            
    return(rootcause)

        
        
ram = pd.read_csv('Ram.csv')

LSTMMODEL = create_model()













    