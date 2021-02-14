# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:12:41 2020

@author: umerg
"""

#Distances are getting Nan Values
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



predicted_label=[]


#print("Test Data: ",testData, "\nPredicted Data: ",predicted)
def similarity_Measure(train,test,distance_name):
    similarities=[]
    similarity_vector=[]
    sum=0
    for i in range(len(test)):
        for j in range(len(train)):
            #print('Value of k:',len(train.iloc[i]))
            for k in range(len(train.iloc[i])):
                if distance_name=='euclidean':
                    sum=sum+distance.euclidean(test.iloc[i][k],train.iloc[j][k])
                elif distance_name=='mahnattan':
                    sum=sum+distance.cityblock(test.iloc[i][k],train.iloc[j][k])
                elif distance_name=='minkowski':
                    sum=sum+distance.minkowski(test.iloc[i][k],train.iloc[j][k])
                elif distance_name=='minkowski':
                    sum=sum+distance.hamming(test.iloc[i][k],train.iloc[j][k])
            similarity_vector.append(sum)
            sum=0
        #print(similarity_vector)
        similarities.append(similarity_vector)
        similarity_vector=[]
    return similarities



def retrieveCases(trainData,K):
    retrieved_cases = []
    for i in range(K):
        retrieved_cases.append(trainData.iloc[i])
    
    return retrieved_cases


def labelPredictor(retrievedData):
    label=[0,0,0,0,0,0,0]
    for i in range(len(retrievedData)):
        index=int(retrievedData.iloc[i]["class_type"])
        label[index-1]=label[index-1]+1
        
    maxLabel=0
    for i in range(len(label)):
        if label[i]>maxLabel:
            maxLabel=i+1
            
    return maxLabel

def CBR(K,train,s):
    predicted_label=[]
    #print("Length of S: ",len(s))
    for i in range(len(s)):
        train['Distance']=pd.DataFrame(s[i])
        train.sort_values("Distance", axis = 0, ascending = True, inplace = True, na_position ='last') 
        train.drop('Distance',axis=1)
        
        retrievedData=retrieveCases(train,K)
        retrievedData=pd.DataFrame(retrievedData)
        #print("Retrieved Case for K: ",K)
        #print(retrievedData)
        #print(test)
        predicted_label.append(labelPredictor(retrievedData))
    return predicted_label 
        #print(train.head(20)) 
        #print(test.iloc[i])
 

def tryWithCustomSplit(distances_name,split_ration,dataset):
     for i in range(3):

        # Calculating Similarites
        distance_name = distances_name.pop()
        print(distance_name.upper(), "Distance")
        
        
        for j in range(len(split_ration)):
            print("Split:",split_ration[j], 'Split Difference:',(split_ration[j][1]-split_ration[j][0]))
            #Multiple Split Rations  #### Train and Test Split
            train = dataset.iloc[:split_ration[j][0],:]
            test = dataset.iloc[split_ration[j][0]:split_ration[j][1],:]
            
    
            # Spliting test X and test Y    
            test_Y = test['class_type']
            test_X = test.drop('class_type',axis=1)
            
            #Converting test labels to list
            testActualLabel = []
            for i in range(len(test_Y)):
                testActualLabel.append(test_Y.iloc[i])
             
            #print(testActualLabel)
        
        
            s=similarity_Measure(train,test,distance_name)
            K=int(split_ration[j][0]/2)
            if(K>30):
                K=50
            for i in range(K):
                predicted=[]
                predicted_label=CBR(i+1,train,s)
                predicted = pd.DataFrame(predicted_label)
                #print("Predicted: ",predicted_label, "Actual: ",testActualLabel)
                print("Accuracy For K: ",i+1,"  is ",accuracy_score(testActualLabel,predicted))        
                
                
def tryWithTrainTestSplit(distances_name,dataset):
    
    for i in range(3):
        # Calculating Similarites
        distance_name = distances_name.pop()
        print(distance_name.upper(), "Distance")
        
        test_ratio = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
        for i in range(len(test_ratio)):
            
            print('Test Ratio:',test_ratio[i])
            #Split dataset into train and test
            train, test = train_test_split(dataset,test_size=test_ratio[i], random_state=1)
            # Spliting test X and test Y    
            test_Y = test['class_type']
            test_X = test.drop('class_type',axis=1)
            
            #Converting test labels to list
            testActualLabel = []
            for i in range(len(test_Y)):
                testActualLabel.append(test_Y.iloc[i])
             
            #print(testActualLabel)
        
        
            s=similarity_Measure(train,test,distance_name)
          
            for i in range(20):
                predicted=[]
                predicted_label=CBR(i+1,train,s)
                predicted = pd.DataFrame(predicted_label)
                #print("Predicted: ",predicted_label, "Actual: ",testActualLabel)
                print("Accuracy For K: ",i+1,"  is ",accuracy_score(testActualLabel,predicted))
            
    


def main(): 
    
    # Preprocessing the dataset
    dataset = pd.read_csv("zoo.csv")
    dataset.head()
    dataset.shape
    dataset=dataset.drop('animal_name', axis=1)
    dataset.shape
    
    distances_name=['euclidean','mahnattan','minkowski']
    split_ratio=[
        
        [10,15],
        [10,20],
        [20,25],         
        [30,35],
        [30,40],
        [40,50],
        [40,70],
        [40,80],
        [50,55],
        [50,70],
        [50,80],
        [60,90],
        [60,101],
        [70,90],
        [70,101],
        [80,101],
        [90,101],  
        ]
    
    #Try 1
    print('Try With Custom Split Function Called')
    #tryWithCustomSplit(distances_name,split_ratio,dataset)
    
    #Try 2
    #print('Try With Train Test Split Function Called')
    tryWithTrainTestSplit(distances_name,dataset)
    
                
if __name__ == "__main__":
    main()       



# k change
# distance change
# split change