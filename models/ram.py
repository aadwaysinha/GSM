import pandas as pd
import random


allFeatures = ['Month', 'Year', 'IsHoliday', 'Quantity', 'ProfitToCompany',
               'InventoryToFactoryDistance', 'TotalWeight', 'ModeOfDelivery', 'RootCause']


def getRandomList(maxLimit):
    row = list()

    for i in range(10000):
        row.append(random.randint(700, maxLimit))

    return row


randomValues = {
                    'Month': ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                    
                    'Year': ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', 
                             '2012', '2013', '2014', '2015', '2016', '2017', '2018'],
                    
                    'IsHoliday': [True, False],
                    
                    'Quantity': getRandomList(10000), 
                    
                    'ProfitToCompany': getRandomList(10000000), #in Rs
                    
                    'InventoryToFactoryDistance': getRandomList(10000), #in KM
                    
                    'TotalWeight': getRandomList(1000000), #in KG
                    
                    'ModeOfDelivery': ['Water', 'Rail', 'Road', 'Airways'],
                    
                    'RootCause': ["Poor Planning",
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
    
    
    
    