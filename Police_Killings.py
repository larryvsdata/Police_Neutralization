# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:06:44 2018

@author: Erman
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class PK():
    
    def __init__(self):
        
        
        self.df=pd.read_csv("police_killings.csv",encoding = "ISO-8859-1")
        self.dfCoded=self.df.drop(['name','month','day','year','streetaddress','city','latitude','longitude'
         ,'state_fp' ,'county_fp' ,'tract_ce' ,'geo_id' ,'county_id','namelsad','lawenforcementagency'
         ,'pop','cause','cause','p_income','h_income','county_income','comp_income','county_bucket','share_white','share_black','share_hispanic'], axis=1).copy()
        self.X=None
        self.y=None
        self.n_estimators=100
        self.clf=RandomForestClassifier(n_estimators=600)
        self.model=self.clf
        self.splitRatio=0.2
        self.trainX=[]
        self.trainY=[]
        self.testX=[]
        self.testY=[]
        self.validationAccuracies=[]
        self.kFold=5
        
        self.models=[]
       
        self.finalAccuracy=0
        
    def cleanseDf(self):
        
        columns=list(self.dfCoded.columns.values)
        
        toBeDeleted=[]
        
        for ind in range(len(self.dfCoded)):
            for column in columns:
                if pd.isnull(self.dfCoded[column][ind]) or self.dfCoded[column][ind]=='Unknown' :
                    toBeDeleted.append(ind)
                    
        toBeDeleted=list(set(toBeDeleted))
        
        self.df.drop(self.df.index[toBeDeleted], inplace=True)
        self.dfCoded.drop(self.dfCoded.index[toBeDeleted], inplace=True)
        self.df.reset_index(inplace=True)
        self.dfCoded.reset_index(inplace=True)
        self.df.reset_index(inplace=True)
        self.df.drop(['index'], axis=1,inplace=True)
        self.dfCoded.drop(['index'], axis=1,inplace=True)
        
        
    def convertToTypes(self):
        
        typeDict={}
        typeDict['age']=int
        typeDict['share_white']=float
        typeDict['share_black']=float
        typeDict['share_hispanic']=float
        typeDict['pov']=float
        
        
        columnList1=['age','share_white','share_black','share_hispanic','pov']
        
        
        for column in columnList1:
            self.df[column]=list(map(typeDict[column], list(self.df[column])))
        
        
        columnList2=['age','pov']
        for column in columnList2:
            self.dfCoded[column]=list(map(typeDict[column], list(self.dfCoded[column])))
            
    

    
        
    def refactorData(self):
        le = preprocessing.LabelEncoder()
        headerList=["gender","raceethnicity","armed","state"]
        
        for column in headerList:
            oneColumn=list(self.df[column])
            self.dfCoded[column]=le.fit_transform(oneColumn)
            
    
            
    def splitTrainTest(self):
        self.y=self.dfCoded[['raceethnicity']].values.tolist()
        self.X=self.dfCoded.drop(['raceethnicity'], axis=1).values.tolist()


        
        
        
    def trainTestSplit(self):

        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio, random_state=42)
    

        
    def trainAndValidate(self):

            self.model.fit(self.trainX,self.trainY)
            validationRatio=1/self.kFold
            
            for validation in range(self.kFold):
               clf=RandomForestClassifier(n_estimators=600)
               self.trainX, self.validateX,self.trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
               clf.fit(self.trainX,self.trainY)
               outcome=clf.predict(self.validateX)
                   
               self.validationAccuracies.append(accuracy_score(outcome,self.validateY))
               self.models.append(clf)
        
    # Choose the model that is the least biased of all validated models.        
            self.model=self.models[self.validationAccuracies.index(max(self.validationAccuracies))]
    
    # Release the memory
            del self.models[:]
        
        

        
  
        
    def test(self):
        self.results=self.model.predict( self.testX)
        self.finalAccuracy=accuracy_score(self.results,self.testY) 
        
    def predictAndScore(self):
        self.results=self.model.predict(self.testX)
        print("Accuracy Score: ", accuracy_score(self.results,self.testY ))
        print("Confusion Matrix: ")
        print( confusion_matrix(self.results,self.testY ))
#        print("Classification Report: ")
#        print( classification_report(self.results,self.testY))
            

        
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii])  
           
    def plot_coefficients(self):
        coef = self.model.feature_importances_
 
         # create plot
        importances = pd.DataFrame({'feature':self.dfCoded.drop(['raceethnicity'], axis=1).columns.values,'importance':np.round(coef,3)})
        importances = importances.sort_values('importance',ascending=True).set_index('feature')
        print( importances)
        importances.plot.barh()

        
        
        

        
        
if __name__ == '__main__':        
        
        pKillings=PK()
        pKillings.cleanseDf()
        pKillings.convertToTypes()
        pKillings.refactorData()
        pKillings.splitTrainTest()
        pKillings.trainTestSplit()
        pKillings.trainAndValidate()
        pKillings.predictAndScore()
        pKillings.plot_coefficients()
        
        
        
#        pKillings.printResults()








        ccc=pKillings.dfCoded
        out=pKillings.y
        X=pKillings.X
        
        
#        
#        for row in X:
#            for element in row:
#                if type(element) ==str:
#                    print(row)
#        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        