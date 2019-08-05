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


  
def create_model(data):                
    #data=pd.read_csv("Keyboard.csv")

    data['date']=0
    for index,row in data.iterrows():
        data.loc[index,'date']=randomDate().date()#("20-01-2018", "23-01-2018").date()
        
    data['date'] = pd.to_datetime(data['date'])
    
    data=data.drop('month',axis=1)
    
    data=data.drop('year',axis=1)
    
    data.columns
    data=data[['date','inventorytofactorydistance', 'isholiday', 'modeofdelivery',
           'profittocompany', 'quantity', 'rootcause', 'totalweight']]
    
    df=data.copy()
    
    df=df.set_index('date')
    
    
    ###### VISUALIZATIONS ##########
#    df.plot(grid=True)
#    
#    import seaborn as sns
#    # Use seaborn style defaults and set the default figure size
#    sns.set(rc={'figure.figsize':(11, 4)})
#    df['profittocompany'].plot(linewidth=0.5)
    
    
    
    ##################################
    
    df1 = pd.get_dummies(df['modeofdelivery'])
    
    df=pd.concat([df,df1],axis=1)
    
    df=df.drop('modeofdelivery',axis=1)
    
    df=df.drop('rail',axis=1)
    
    
    df.columns
    
    
    df=df[['inventorytofactorydistance', 'isholiday', 'profittocompany',
           'quantity', 'totalweight', 'airways', 'road', 'water', 'rootcause']]
    
    
    x= df.iloc[:, 0:8]
    y= df.iloc[:,-1:]
    
    x['isholiday']=x['isholiday'].astype(int) 
    
    y = pd.get_dummies(y['rootcause'])
    y=y.drop('Poor Planning',axis=1)
    
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    x_training_set_scaled = sc_X.fit_transform(x)
    #x_final=pd.DataFrame(x_training_set_scaled)
    #x_final['date']=x['date']
    #x_training_set_scaled['date']=x['date']
    sc_Y = StandardScaler()
    y_training_set_scaled=sc_Y.fit_transform(y)
    
    
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
#            predicted_cause = classifier.predict(X_test)
#            predicted_cause = sc.inverse_transform(predicted_cause)
    
    return classifier
    
    
    
def retMAX(predicted_cause):   
    print(predicted_cause)     
    df_final=pd.DataFrame(predicted_cause)
    rootcause=''
    maxx = -100000
    for i in range(14):
        if abs(df_final.iloc[0][i]) > maxx:
            maxx = abs(df_final.iloc[0][i])
            rootcause =rootCauseDict[i]
            print(maxx)
            print(rootcause)
            
    if(rootcause == ''):
            rootcause = 'Transport Delays'    
    return(rootcause)
    
    

def findRootcause(classifier,features):
    columns=['inventorytofactorydistance','isholiday','modeofdelivery','month','profittocompany','quantity','totalweight','year']
     
    dflist = []
    
    rlist = []
    for key in columns:
        if features[key] is None:
            rlist.append(None)
        else:
            rlist.append(features[key])
    
    dflist.append(rlist)
    
    df = pd.DataFrame(dflist)
    df=np.array(df)
    df=np.reshape(df,(df.shape[0],df.shape[1],1))
    rootcause = classifier.predict(df)
    #will print the root_cause a particular row only
    return(retMAX(rootcause)) #will be of type string
     
     
    #    data=data[['date','inventorytofactoryDistance', 'isholiday', 'modeofdelivery',
         #  'profittocompany', 'quantity', 'rootcause', 'totalweight']] 
     
        
features = {
                'inventorytofactorydistance':2345 , 
                'isholiday': 0,
                'modeofdelivery': 0,
                'month': 11,
                'profittocompany':251, 
                'quantity': 4233,
                'totalweight': 2134,
                'year': 2017
           }     
     

df = pd.read_csv('Ram.csv')
classifier1 = create_model(df)



x = findRootcause(classifier1, features)

t1=['inventorytofactorydistance','isholiday','modeofdelivery','month','profittocompany','quantity','totalweight','year']
t2 = pd.DataFrame(list(features.items()),columns=t1)


dflist = []

rlist = []
for key in t1:
    if features[key] is None:
        rlist.append(None)
    else:
        rlist.append(features[key])

dflist.append(rlist)

df = pd.DataFrame(dflist)
df=np.array(df)
df=np.reshape(df,(df.shape[0],df.shape[1],1))
classifier1.predict(df)


rc = findRootcause(classifier1, features)

    
