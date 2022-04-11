#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:55:05 2022

@author: abelson
"""

import pandas as pd
import numpy as np

#reading csv file
dataset=pd.read_csv('Churn_Modelling.csv') 
#Seperate Independent and dependednt features
X=dataset.iloc[:,3:-1]
y=dataset.iloc[:,-1]

#OHE on categorical features
geography=pd.get_dummies(X['Geography'],drop_first=True) 
gender=pd.get_dummies(X['Gender'],drop_first=True)

#Add OHE back to X
X=pd.concat([X,geography,gender], axis=1) 

#drop categorical variables
X.drop(['Geography','Gender'],axis=1, inplace=True) 

#train test split to avoid overfitiing and data leakage
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

#fit transform standard scalar standard normal distributions
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu,sigmoid 

#create a function for building neural network model
def create_model(layers,activation):
    #assign variable to NN sequential
    model=Sequential() 
    for i,nodes in enumerate(layers):
        #if layer being built is 1st layer, input layer should be number of independent features
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        #if layer being buillt is middle layer
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    #output layer
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    
    #compile all the layers
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

#call NN building fuction
model= KerasClassifier(build_fn=create_model,verbose=0)

#determine layers
layers=[[20],[40,20],[45,30,15]]
activations=['sigmoid','relu']
param_grid=dict(layers=layers,activation=activations, batch_size=[128,256],epochs=[30])
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=5)

grid_result=grid.fit(X_train,y_train)
results=[grid_result.best_score_,grid_result.best_params_]
results







