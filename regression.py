# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:42:05 2017

@author: Young
"""
import numpy as np

def load_dataset(fileName):
    """
    从文件读取训练数据（X,Y）,文件中数据项以‘\t’间隔
    """
    dataset = []
    fr = open(fileName)
    for line in fr:
        dataset.append(line.split('\t').strip())
    dataset = np.array(dataset)        
    return dataset[:,0:-1], dataset[:,-1]

if __name__ == "__main__":
    X, Y = load_dataset('ex0.txt')