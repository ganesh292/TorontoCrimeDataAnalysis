# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:27:41 2019

@author: Kunal Taneja
"""

import numpy as np
import pandas as pd
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
dfTrain = pd.DataFrame(X_train.reshape(60000,784))
dfTrain['Label'] = y_train.reshape(60000,1)
dfTrain = dfTrain.head(10000)

grouped = dfTrain.groupby('Label')


def maha_dist(X, features):
    
    if(len(features) == 1):
        
        for i in features:
            x = X.iloc[:,i].values
            
        mu = x.mean()
        x_var = x.var()
            
        if(x_var == 0):
            inv = x_var
        else:
            inv = 1/x_var
                
        x = (x-mu)
        #x = np.reshape(x, newshape=(np.shape(x)[0], 1, np.shape(x)[1]))
            
        #inv = np.reshape(inv, newshape=(1, np.shape(inv)[0], np.shape(inv)[1]))
        
        dist = np.dot(np.dot(x.T, inv), x)
        
    else:
        list1 = []
        for i in features:
            temp = X.iloc[:,i].values
            list1.append(temp)
            
        x = np.array(list1)
        mu = x.mean()
        
        x_cov = np.cov(x)
        inv = np.linalg.pinv(x_cov)

        x = (x - mu)
        z = np.reshape(x, newshape=(np.shape(x)[1], np.shape(x)[0], 1))
        x = np.reshape(x, newshape=(np.shape(x)[0], 1, np.shape(x)[1]))
            
        inv = np.reshape(inv, newshape=(1, np.shape(inv)[0], np.shape(inv)[1]))
        
        dist = np.matmul(np.matmul(x.T, inv), z)
#         print (dist.shape)
        
    return dist.mean()



def search(d):
    fw_list = []
    bw_list = np.arange(784)
    all_features = np.arange(784)
    temp_bc_list = np.arange(784)
    while(len(fw_list)!=d and len(bw_list)!=d):
        #force running SFS

        distance_classes = []
        for j in range(0,10):
            a = grouped.get_group(j)
            distance = []
            for i in all_features:
                features = [i]
                features.extend(fw_list)
                distance_val = maha_dist(a,features) #MAIN RUN

                distance.append(distance_val)
            distance_classes.append(distance)
        distance_df = pd.DataFrame(distance_classes)
        distances_sum = distance_df.sum(axis=0)
        feature_best = distances_sum.idxmax(axis=0)
        feature_worst = distances_sum.idxmin(axis=0)
        feature_best = all_features[feature_best]
        feature_worst= all_features[feature_worst]

        if(feature_best not in fw_list and feature_best in bw_list):
            fw_list.append(feature_best)
            #if(feature_worst not in fw_list and feature_worst in bw_list):
            #    np.delete(bw_list, np.where(bw_list == feature_worst))
        print("done with SFS")
        print("best feature is:",feature_best)
        print("worst feature is:",feature_worst)
        print("Foward List is:",fw_list)

        #force running SBS

        distance_classes = []
        for j in range(0,10):
            a = grouped.get_group(j)
            distance = []
            for i in all_features:
                features = np.delete(temp_bc_list, np.where(temp_bc_list == i))

                distance_val = maha_dist(a,features) #MAIN RUN

                temp_bc_list = bw_list

                distance.append(distance_val)
            distance_classes.append(distance)
        distance_df = pd.DataFrame(distance_classes)
        distances_sum = distance_df.sum(axis=0)
        feature_best_bw = distances_sum.idxmin(axis=0)
        feature_worst_bw = distances_sum.idxmax(axis=0)
        feature_best_bw = all_features[feature_best_bw]
        feature_worst_bw= all_features[feature_worst_bw]

        if(feature_worst_bw in bw_list and feature_worst_bw not in fw_list):
            bw_list=np.delete(bw_list, np.where(bw_list == feature_worst_bw))


        print("done with SBS")
        print("best feature is:",feature_best_bw)
        print("worst feature is:",feature_worst_bw)
        print("Backward List is:",bw_list)
        
        all_features = np.delete(all_features, np.where(all_features == feature_worst_bw))
        all_features = np.delete(all_features, np.where(all_features == feature_best))
        
        print("All feature List is:",all_features)
    
    
    print('best features extracted are:')  
    if(len(fw_list==d)):
        print(fw_list)
    else:
        print(bw_list)
        
        
        #if(feature_best not in fw_list and feature_best in bw_list):
        #    fw_list.append(feature_best)
        #if(feature_worst not in fw_list and feature_worst in bw_list):
        #    np.delete(bw_list, np.where(bw_list == feature_worst))
def main():
    search(5)
    
if __name__ == '__main__':
    main()