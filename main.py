# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:06:34 2022

"""

#### IMPORTS ####

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


#### READING FILES ####

# Read Train data
df_orders = pd.read_csv("orders.csv", sep=";")
df_products = pd.read_csv("product_attributes.csv", sep=",")
df_dists = pd.read_csv("cities_data.csv", sep=";")

# Read Test data
df_test = pd.read_csv("test.csv", sep=";")


#### PREPROCESSING ####

# merging orders an products
def dropProductId(df_orders,df_products):
    df_orderswm = pd.merge(df_orders, df_products, on='product_id')
    return df_orderswm

df_orderswm = dropProductId(df_orders, df_products)
df_testwm = dropProductId(df_test, df_products)

# BCN -> Barcelona
# ATHENES -> Athenes
def BCNATHENAS(df):
    df.loc[df['origin_port'] == "BCN","origin_port"] = "Barcelona"
    df.loc[df['origin_port'] == "ATHENAS","origin_port"] = "Athens"
    return df

df_orderswm = BCNATHENAS(df_orderswm)
df_testwm = BCNATHENAS(df_testwm)

# binary encoding
df_orderswm['late_order'] = df_orderswm['late_order'].replace( [True, False], [1,0])

# one hot encoding
onehot_encoder = ce.OneHotEncoder(cols=['origin_port', '3pl', 'customs_procedures', 'logistic_hub', 'customer','material_handling','product_id'])
df_order_encoded = onehot_encoder.fit_transform(df_orderswm)

onehot_encoder.fit(df_orderswm.drop(axis = "columns", columns=["late_order"]))
df_test_encoded = onehot_encoder.transform(df_testwm)


df_order_encoded.drop('order_id', axis=1, inplace = True)
df_order_encoded.drop(["weight"], axis = 1)


#### TRAINING ####
           
# train test split
                    
matriu_x= df_order_encoded.loc[:,:]
matriu_x = matriu_x.drop(["late_order"], axis = 1)
matriu_y= df_order_encoded.loc[:, ["late_order"]]

X_train, X_test, y_train, y_test = train_test_split(matriu_x, 
                                                    matriu_y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# standardization

X_mean = matriu_x["units"].mean()
X_std = matriu_x["units"].std()

matriu_x["units"] = (matriu_x["units"] - X_mean)/X_std
df_test_encoded["units"] = (df_test_encoded["units"] - X_mean)/X_std

X_mean = matriu_x["weight"].mean()
X_std = matriu_x["weight"].std()

matriu_x["weight"] = (matriu_x["weight"] - X_mean)/X_std
df_test_encoded["weight"] = (df_test_encoded["weight"] - X_mean)/X_std




#%%
# grid searchs

grids_searchs = []

kfold = model_selection.KFold(n_splits=5,shuffle=True, random_state=0)

linearsvc = {"model": LinearSVC(), "params": {'penalty': ["l1", "l2"], 'C':[0,0.01,0.1,10,100], "max_iter": [1000,2000,5000]}}

svc = {"model": SVC(), "params": {'C':[0,0.01,0.1,10,100], 'gamma': [0.01,0.01,0.1,10,100], 'kernel': ["rbf", "linear"]}}

lr = {"model": linear_model.LogisticRegression(), "params": {'penalty': ["l1", "l2", "elasticnet"], 'C':[0,0.01,0.1,10,100], 'max_iter':[500, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}}

rf = {"model": RandomForestClassifier(), "params": {"n_estimators": [100, 500, 1000], "n_jobs" : [42, -1], "criterion": ["gini", "entropy"]}}

lr2 = {"model": linear_model.LogisticRegression(), "params": {'class_weight':['balanced'],'penalty': ["l2"], 'C':[10,30,100], 'max_iter':[1000]}}

mlpC = {"model": MLPClassifier(),"params":{"activation":["relu"], "hidden_layer_sizes": [(5,1000)]}}

for model in [lr2, linearsvc, svc, lr, rf, mlpC]:
    clf_ = GridSearchCV(model["model"], model["params"], scoring="f1", cv=kfold)
    clf_.fit(X_train, y_train)
    grids_searchs.append(clf_)

# %%

#### PREDICTION ####

from sklearn.metrics import roc_curve, auc


b = MLPClassifier(alpha=0.001)
b.fit(matriu_x, matriu_y)

#test_F = pd.read_csv('test2.csv', sep=",")
id_test = pd.DataFrame(df_test_encoded["order_id"])
c = df_test_encoded.drop(['order_id'],axis=1)
yhatF = b.predict_proba(c)
id_test['late_order'] = yhatF[:,0]

y_pred_prob = b.predict_proba(matriu_x)
fpr, tpr, thresholds = roc_curve(matriu_y, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

# %%


nou = pd.merge(df_test, id_test, on='order_id', how = "left")

nou['late_order'] = nou['late_order'].fillna(0.5)

nou['late_order'] = 1 - nou["late_order"]

nou2= nou.loc[:,["order_id", "late_order"]]

nou2.to_csv('primera_entrega.csv',index=False)