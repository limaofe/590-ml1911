# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:09:25 2021

@author: Mao Li
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

class Data:
    
    # Read the json data
    def __init__(self, filename, limit_gt18=0):
        self.data=pd.read_json(filename)
        if limit_gt18==0:
            return
        else:
            self.data=self.data[self.data['x']<18]
        
    # Normalize inputs (features) and outputs as needed using “standard scalar”
    def normalize(self):
        x=self.data['x'].values
        y=self.data['y'].values
        x_mean=np.mean(x)
        x_std=np.std(x)
        x_list=[]
        for i in x:
            i=(i-x_mean)/x_std
            x_list.append(i)
        self.data['normalized_x']=x_list
        y_mean=np.mean(y)
        y_std=np.std(y)
        y_list=[]
        for i in y:
            i=(i-y_mean)/y_std
            y_list.append(i)
        self.data['normalized_y']=y_list
        self.x_mean=x_mean
        self.x_std=x_std
        self.y_mean=y_mean 
        self.y_std=y_std
        return
    
    # Visualize the data as a scatter plot
    def visualize(self):
        self.x=self.data['x'].values
        self.y=self.data['y'].values
        self.l=self.data['is_adult'].values
        plt.scatter(self.x, self.y)
        plt.xlabel('Age')
        plt.ylabel('Weight')
        plt.title('Visualization')
        plt.show()        
        
    def shape(self):
        return self.data.shape
        
    # break the data into 80% training 20% test            
    def partition(self, test_size=0.2):
        features=self.data['x'].values
        features_n=self.data['normalized_x'].values
        targets=self.data['y'].values
        targets_n=self.data['normalized_y'].values
        labels=self.data['is_adult']
        # split the original data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(features, targets, test_size=test_size, random_state=1911)
        # split the normalized data
        self.train_x_n, self.test_x_n, self.train_y_n, self.test_y_n = train_test_split(features_n, targets_n, test_size=test_size, random_state=1911)
        # split the original data with the label
        self.train_y_logi, self.test_y_logi, self.train_label, self.test_label=train_test_split(targets, labels, test_size=test_size, random_state=1911)
        
class LinerRegression:
    
    def __init__(self, train_x, train_y, test_x, test_y):
        self.w=0
        self.b=0
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y 
        
    def loss(self, p):
        y_pred=np.dot(self.train_x, p[0])+p[1]
        e=self.train_y-y_pred
        se=np.power(e,2)
        rse=np.sqrt(np.sum(se))
        rmse=rse/self.train_y.shape[0]
        return rmse
        
    def optimize(self):
        self.p=minimize(self.loss, [self.w, self.b], method='nelder-mead')
    
    def pred(self):
        self.optimize()
        y_pred=np.dot(self.test_x, self.p.x[0])+self.p.x[1]
        self.w=self.p.x[0]
        self.b=self.p.x[1]
        return y_pred
    
class LogisticRegression:
    def __init__(self, train_x, train_y, test_x, test_y, method):
        self.p0=0
        self.p1=0
        self.p2=0
        self.p3=0
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y 
        self.method=method
    
    # p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.00001))))    
    def fit(self, input, p):
        res=[]
        for i in range(len(input)):
            y=p[0]/(1+np.e**((p[2]-input[i])/p[1]))+p[3]
            #y=p[0]+p[1]*(1.0/1.0+np.exp(-(input[i]-p[2])/(p[3]+0.00001)))
            res.append(y)
        #y=np.dot(dataset, p[0])+p[1]
        #res=1/(1+np.exp(-y))
        return res
    
    def loss(self, p):
        e=self.train_y-self.fit(self.train_x, p)
        se=np.power(e,2)
        rse=np.sqrt(np.sum(se))
        rmse=rse/self.train_y.shape[0]
        return rmse
        
    def optimize(self):
        self.p=minimize(self.loss, [self.p0, self.p1, self.p2, self.p3], method='Powell')
        
    def pred(self):
        self.optimize()
        y_pred=[]
        if self.method=='c':
            prob=self.fit(self.test_x, self.p.x)
            for i in prob:
                if i<=0.5:
                    i=0
                if i>0.5:
                    i=1
                y_pred.append(i)
        else:
            y_pred=self.fit(self.test_x, self.p.x)
        self.p0=self.p.x[0]
        self.p1=self.p.x[1]
        self.p2=self.p.x[2]
        self.p3=self.p.x[3]
        self.p=self.p.x
        return y_pred
        
    
    
if __name__ == '__main__':
    # linear regression
    # read the dataset that the age is less than 18
    data=Data('weight.json',1)        
    data.shape()
    data.normalize()
    data.shape()
    data.visualize()
    data.partition()

    lr=LinerRegression(data.train_x_n, data.train_y_n, data.test_x_n, data.test_y_n, 1)
    y_pred=lr.pred()
    plt.scatter(data.x, data.y)
    # inverting the normalization
    x_i=data.train_x_n*data.x_std+data.x_mean
    y_i=(data.train_x_n*lr.w+lr.b)*data.y_std+data.y_mean
    plt.plot(x_i, y_i, color='r')
    plt.xlabel('Age')
    plt.ylabel('Weight')
    plt.title('Liner Regression')
    plt.show()
    print('The slope for the linear regression is ', lr.w, ' and the intercept is ', lr.b, ' .', sep='')
    
    # logistic regression
    # read the whole dataset for the logistic regression
    data=Data('weight.json')        
    data.shape()
    data.normalize()
    data.shape()
    data.visualize()
    data.partition()
    
    logi=LogisticRegression(data.train_x_n, data.train_y_n, data.test_x_n, data.test_y_n, 'r')
    y_pred=logi.pred()  
    plt.scatter(data.x, data.y)	
    x_i=data.train_x_n*data.x_std+data.x_mean
    y=logi.fit(data.train_x_n, logi.p)
    y_i=np.dot(logi.fit(data.train_x_n, logi.p),data.y_std)+data.y_mean
    def f(x):
        
    plt.plot(x_i, y_i, color='r')
    plt.xlabel('Age')
    plt.ylabel('Weight')
    plt.title('Logistic Regression for regression')
    plt.show()
    print('The parameter of the logistic regression for regression is ', logi.p)
    
    logi=LogisticRegression(data.train_y_logi, data.train_label, data.test_y_logi, data.test_label, 'c')
    y_pred=logi.pred()  
    plt.scatter(data.y, data.l)	
    x_i=data.train_y.tolist()
    y_i=logi.fit(data.train_y_logi, logi.p)
    plt.scatter(x_i, y_i, color='r')
    plt.xlabel('Weight')
    plt.ylabel('Is Adult')
    plt.title('Logistic Regression for Classification')
    plt.show()
    print('The parameter of the logistic regression for classification is ', logi.p)
