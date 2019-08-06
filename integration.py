import pickle
import pandas as pd
import numpy as np
import random
# import keras

#STEP 1
#load all pickle files and store in a dictionary 
#{ partName => classifier }

allParts = ['hdd']
#, 'screen', 'ram', 'laptop', 'server', 'mouse', 'keyboard'

def loadModels():
    models = dict()
    print("IN LOAD MODELS")
    for part in allParts:
        loadedModel = pickle.load(open(part + 'stored.sav', 'rb'))
        models[part] = loadedModel
        print(part + " loaded")
    print("ALL MODELS ARE READY")
    return models



#STEP 2
#Create a function which takes classifier and an input dictionary as the inputs and return the root cause
#These functions are already present in RAM.py

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


def retMAX(predicted_cause):   
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
    columns=['inventorytofactorydistance','isholiday','modeofdelivery','month','profittocompany','quantity','manufacturingtime','year', 'daystilldelivery']
     
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



def generateRandomInputs():
    features = {
                    'inventorytofactorydistance': random.randint(500, 10000) // random.randint(10, 1000), 
                    'isholiday': random.randint(0, 1),
                    'modeofdelivery': random.randint(1, 4),
                    'month': random.randint(1, 12),
                    'profittocompany': random.randint(1000, 10000), 
                    'quantity': random.randint(10, 1000),
                    'manufacturingtime': random.randint(2, 500),
                    'daystilldelivery': random.randint(2, 500),
                    'year': random.randint(2000, 2018)
               }
    print("Random input is generated")
    return features


def getRootCause(classifier, part):
    features = generateRandomInputs()
    return findRootcause(classifier, features)


#STEP 3
#Map all the root causes to certain actions and return the action for a particular root cause 


def findAction(root):
    action={'Strike of workers':'increase wage','Poor Lead time calculation':'recalculate supply and reorder time','electricty stoppage':'install more generators','Transport Delays':'move inventory','Demand Variation':'recalculate demand curve','Supply Shortages and logistical uncertainties':'improve logistics','excessive machine stoppage':'increase staff communication','Poor Planning':'improve project management','Raw material low':'increase production','Poor inventory control':'monitor inventory','faulty plant layout':'Look into the issue and fix plant','material wastage due to over-feeding':'increase staff communication','Factory shutdown':'move inventory','Huge backlog of orders':'increase production','Financial problems of company leading to interrupted supplies':'limit orders'}
    return action.get(root) 



MODELS = loadModels()
print("ROOT CAUSE: " + getRootCause(MODELS['hdd'], 'hdd'))