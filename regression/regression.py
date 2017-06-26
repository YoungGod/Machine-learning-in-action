# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:42:05 2017

@author: Young
"""
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(fileName):
    """
    从文件读取训练数据（X,Y）,文件中数据项以‘\t’间隔
    """
    dataset = []
    fr = open(fileName)
    for line in fr:
        dataset.append([float(x.strip()) for x in line.split('\t')])
    dataset = np.array(dataset)        
    return dataset[:,0:-1], dataset[:,-1]

def standard_reg(X,Y):
    """
    Y = w0*x0 + w1*x1 + ... + wn*xn + e
    Y = w*X + e
    w = argmin ||Y-w*X||
    解析解：w = inv(X.T*X)*(X.T*Y)
            w = pinv(X)*Y
    """
    from numpy.linalg import pinv
    w = np.dot(pinv(X),Y)
    return w
    
    
if __name__ == "__main__":
    X, Y = load_dataset('ex0.txt')
    fig, ax = plt.subplots()
    ax.scatter(X[:,-1],Y)
    w = standard_reg(X,Y)
    Y_pred = X[:,0]*w[0] + X[:,1]*w[1]
    ax.scatter(X[:,-1],Y_pred)
    