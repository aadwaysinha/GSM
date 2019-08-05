import pandas as pd
import random


allFeatures = ['month', 'year', 'isholiday', 'quantity', 'profittocompany',
               'inventorytofactorydistance', 'totalweight', 'modeofdelivery', 'rootcause']


def getRandomList(maxLimit):
    row = list()

    for i in range(10000):
        row.append(random.randint(700, maxLimit)//random.randint(10, 1000))

    return row


randomValues = {
                    'month': ["January","February","March","April","May","June","July","August","September","October","November","December"],
                    
                    'year': ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', 
                             '2012', '2013', '2014', '2015', '2016', '2017', '2018'],
                    
                    'isholiday': [True, False],
                    
                    'quantity': getRandomList(10000), 
                    
                    'profittocompany': getRandomList(10000000), #in Rs
                    
                    'inventorytofactorydistance': getRandomList(10000), #in KM
                    
                    'totalweight': getRandomList(1000000), #in KG
                    
                    'modeofdelivery': ['water', 'rail', 'road', 'airways'],
                    
                    'rootcause': ["Poor Planning",
                                 "Strike of workers",
                                 "Poor Lead time calculation",
                                 "Poor inventory control",
                                 "faulty plant layout",
                                 "excessive machine stoppage",
                                 "electricty stoppage",
                                 "Raw material low",
                                 "material wastage due to over-feeding",
                                 "Demand Variation",
                                 "Huge backlog of orders",
                                 "Supply Shortages and logistical uncertainties",
                                 "Factory shutdown",
                                 "Financial problems of company leading to interrupted supplies",
                                 "Transport Delays"]
               }


def appendRow():
    row = dict()

    for feature in allFeatures:
        row[feature] = random.choice(randomValues[feature])

    return row


allParts = ['HDD', 'Screen', 'Ram', 'Laptop', 'Server', 'Mouse', 'Keyboard']


for part in allParts:
    rowList = []
    
    for i in range(500):
        rowList.append(appendRow())
    
    df = pd.DataFrame(rowList)
    
    filename = part + '.csv'
    df.to_csv(filename, index = False)    




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
    
    

allParts = ['HDD', 'Screen', 'Ram', 'Laptop', 'Server', 'Mouse', 'Keyboard']

dataSets =  {
                'hdd': pd.read_csv('HDD.csv'),
                'screen': pd.read_csv('Screen.csv'),
                'ram': pd.read_csv('Ram.csv'),
                'laptop': pd.read_csv('Laptop.csv'),
                'server': pd.read_csv('Server.csv'),
                'mouse': pd.read_csv('Mouse.csv'),
                'keyboard': pd.read_csv('Keyboard.csv')
            }



def preComputeModels():
    models = {}
    for dfName in dataSets:
        df = dataSets[dfName]
        classifier = create_model(df)
        models[dfName] = classifier
    
    return models

models = preComputeModels()

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

#Test for a single model

df111 = pd.read_csv('Screen.csv')

c111 = create_model(df111)

import pickle
pickle.dump(c111, open('testPickle.sav', 'wb'))

import dill as pickle
with open('name_model.pkl', 'wb') as file:
    pickle.dump(c111, file)



rc = findRootcause(models['ram'], features)


#Savinf model to a pickle file
import pickle

def storeModels():
    for modelName in models:
        classifier = models[modelName]
        rc = findRootcause(models['ram'], features)
        print("ROOT CAUSE: ", rc)
        print(type(classifier))
        fileName = modelName + 'stored.sav'
        pickle.dump(classifier, open(fileName, 'wb'))
        
storeModels()


# to find the rootcause for any input: rc = findRootcause(classifier1, features)


import pandas as pd
import numpy as np












def generateRandomInputs():
    features = {
                    'inventorytofactorydistance': random.randint(500, 10000) // random.randint(10, 1000), 
                    'isholiday': random.randint(0, 1),
                    'modeofdelivery': random.randint(0, 4),
                    'month': random.randint(0, 12),
                    'profittocompany': random.randint(1000, 10000000), 
                    'quantity': random.randint(100, 10000),
                    'totalweight': random.randint(100, 100000),
                    'year': random.randint(2000, 2018)
               }
    return features


f= generateRandomInputs()




#import json
#dataSets =  {
#                'hdd': pd.read_csv('HDD.csv'),
#                'screen': pd.read_csv('Screen.csv'),
#                'ram': pd.read_csv('Ram.csv'),
#                'laptop': pd.read_csv('Laptop.csv'),
#                'server': pd.read_csv('Server.csv'),
#                'mouse': pd.read_csv('Mouse.csv'),
#                'keyboard': pd.read_csv('Keyboard.csv')
#            }  
#    
#def bringDataSet(requirements):
#
#    keyword = requirements['searchKeyword'].lower()
#    category = requirements['searchCategory'].lower()
#
#    newTable = dict()
#    rowList = []
#    
#    if category in dataSets:
#        df = dataSets[category] 
#        for index, row in df.iterrows():
#            rowList.append(list(row.values))
#        if len(rowList) > 1:
#            newTable[category] = rowList
#        
#    else:
#        for dfName in dataSets:
#            df = dataSets[dfName]
#            partType = dfName
#            for index, row in df.iterrows():
#                if(row[category].lower() == keyword.lower()):
#                    rowList.append(list(row.values))
#            if len(rowList):
#                newTable[partType] = rowList
#                
#
#    jsonedDF = json.dumps(newTable)
#    return jsonedDF
#
#
#
#requirement = {
#                'searchKeyword': 'Dec',
#                'searchCategory': 'Month' 
#               }
#
#x = json.loads(bringDataSet(requirement))
#    
#
#














#dataSets =  {
#                'hdd': pd.read_csv('HDD.csv'),
#                'screen': pd.read_csv('Screen.csv'),
#                'ram': pd.read_csv('Ram.csv'),
#                'laptop': pd.read_csv('Laptop.csv'),
#                'server': pd.read_csv('Server.csv'),
#                'mouse': pd.read_csv('Mouse.csv'),
#                'keyboard': pd.read_csv('Keyboard.csv')
#            }        
#
#import json
#
#def bringDataSet(requirements):
#
#    keyword = requirements['searchKeyword']
#    category = requirements['searchCategory']
#
##    keyword = keyword
##    category = category
#
#    newTable = dict()
#    rowList = []
#    
#    if category in dataSets: 
#        for index, row in dataSets[category].itterrows():
#            rowList.append(list(row.values))
#        if len(rowList) > 1:
#            newTable[category] = rowList
#        
#    else:
#        for dfName in dataSets:
#            df = dataSets[dfName]
#            partType = dfName
#            for index, row in df.iterrows():
#                if(row[category].lower() == keyword.lower()):
#                    rowList.append(list(row.values))
#            if len(rowList):
#                newTable[partType] = rowList
#                
#
#    jsonedDF = json.dumps(newTable)
#    return jsonedDF
#    
#
#
#
#
#requirement = {
#                'searchKeyword': 'Jan',
#                'searchCategory': 'Month' 
#               }
#
#x = json.loads(bringDataSet(requirement))
#    
#for part in x:
#    print("Part: ", part)
#    for col in x[part]:
#        print(col)





    