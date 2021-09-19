# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:09:08 2021

@author: ag965
"""
import pandas as pd
import numpy as np
#from sklearn.decomposition import PCA as pca
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
# reading csv files
data =  pd.read_csv('iris.data', sep=",")
data_1 =np.array(data)
new=[[5.1,3.5,1.4,0.2,'Iris-setosa']]
axes=Axes3D(plt.figure(1,figsize=(8,6)),elev=15,azim=30)
for i in range(0,len(data_1)):
    new.append((data_1[i]))
new=np.array(new)
X=new[:,0:4]
X=np.c_[np.ones(len(X)),X]
Y=new[:,4]
y_classes=list(set(list(Y)))
flag=0
def lable(Y,num,y_lable,y_classes):
    for i in range(len(Y)):
        if Y[i]==y_classes[num]:
            y_lable.append(1)
        else:
            y_lable.append(-1)
    
    return y_lable
y_lable=lable(Y,0,[],y_classes)
n=1
def classify(X,y_lable):
    w=np.zeros(len(X.T))
    weight=[]
    i=0
    while i<=len(X)-1:
        if ((y_lable[i]==1) and ((np.dot(w,X[i])))>=0) or ((y_lable[i]==-1) and ((np.dot(w,X[i])))<0):
            i+=1
        else :
            #pdb.set_trace()
            temp=i
            if y_lable[i]==1:
                w=w+X[i]
            else:
                w=w-X[i]
            i=0
    weight.append(w)
    return weight
w=classify(X,y_lable)
col={0:'r',1:'g',2:'b'}
def diff_lable(Y):
    col_y=[]
    for i in range(len(Y)):
        for j in range(3):            
            if Y[i]==y_classes[j]:
                col_y.append(j)
    return col_y
# x=np.array(X[:,1])
# y=np.array(X[:,2])
# m,n=np.meshgrid(x,y)
# z=-(w[2]*n-w[1]*m-w[0])/(w[3])

# for i in range(3):
#     idx=np.where(Y==i)
#     axes.scatter(X[idx,1],X[idx,2],X[idx,3])
#     labels=y_classes[i]

        
# col_y=diff_lable(Y)    
# color=[]
# for i in range(len(Y)):
#     color.append(col[col_y[i]])
# plt.scatter(X[:,1],X[:,2],c=color)
# print(data)