import pandas as pd
import numpy as np
import json

#importing all datasets
dataSets =  {
                'hdd': pd.read_csv('datasets/HDD.csv'),
                'screen': pd.read_csv('datasets/Screen.csv'),
                'ram': pd.read_csv('datasets/Ram.csv'),
                'laptop': pd.read_csv('datasets/Laptop.csv'),
                'server': pd.read_csv('datasets/Server.csv'),
                'mouse': pd.read_csv('datasets/Mouse.csv'),
                'keyboard': pd.read_csv('datasets/Keyboard.csv')
            }


def convertDFtoJsonTable(DF):
    return 0


def bringDataSet(requirements):

    keyword = requirements['searchKeyword']
    category = requirements['searchCategory']

    if type(keyword) == str:
        keyword = keyword.lower()
    if type(category) == str: 
        category = category.lower()

    newTable = dict()
    rowList = []
    
    if category in dataSets:
        df = dataSets[category] 
        for index, row in df.iterrows():
            rowList.append(list(row.values))
        if len(rowList) > 1:
            newTable[category] = rowList
        
    else:
        for dfName in dataSets:
            df = dataSets[dfName]
            partType = dfName
            for index, row in df.iterrows():
                if((type(row[category]) == str) and ((row[category].lower() == keyword.lower()) or (keyword.lower() in row[category].lower()))): #for string
                    rowList.append(list(row.values))
                elif((type(row[category] == int) and keyword.isnumeric()) and (int(row[category]) == int(keyword))): #for integer
                    rowList.append(list(row.values))
                elif((type(row[category]) == bool and keyword in ['true', 'false'])):
                    holiday = keyword
                    if(holiday == 'true'):
                        holiday = True
                    elif(holiday == 'false'):
                        holiday = False
                    if(row[category] == holiday):
                        rowList.append(list(row.values))
            if len(rowList):
                newTable[partType] = rowList
                

    jsonedDF = json.dumps(newTable)
    return jsonedDF
    

