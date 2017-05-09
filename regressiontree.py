# -*- coding: utf-8 -*-
"""
Created on Tue May 09 10:48:15 2017

@author: Young
"""

import numpy as np

def load_dataset(filename):
    data_set = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)
        data_set.append(flt_line)
    return np.array(data_set)

def bin_split(data_set, feature, value):
    data_set = np.array(data_set)
#    mat0 = data_set[np.nonzero(data_set[:, feature] > value)]
    right_set = data_set[np.where(data_set[:,feature] > value)]
#    print data_set[np.nonzero(data_set[:, feature] > value)]
    left_set = data_set[np.nonzero(data_set[:, feature] <= value)]
    return right_set, left_set

def reg_leaf(data_set):
    return np.mean(data_set[:,-1])

def reg_err(data_set):
    return np.var(data_set[:,-1]) * np.shape(data_set)[0]

def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    """
    功能：选取最佳划分(特征，取值)
    根据叶节点回归方法leaf_type，策略err_type，以及选项ops，选取最优划分
    """
    tol_s = ops[0]; tol_n = ops[1]
    if len(np.unique(data_set[:,-1])) == 1:
        return None, leaf_type(data_set)
    
    best_err = np.inf; best_feat = 0; best_value = 0
    for i_feat in xrange(data_set.shape[1]-1):
        for value in np.unique(data_set[:,i_feat]):
            right_set, left_set = bin_split(data_set, i_feat, value)
            if right_set.shape[0] < tol_n or left_set.shape[0] < tol_n:  # 最小划分集合限定
                continue
            new_err = err_type(right_set)+err_type(left_set)
            if new_err < best_err:
                best_feat = i_feat
                best_value = value
                best_err = new_err
#    print best_feat, best_value, best_err, err_type(data_set)
    if err_type(data_set) - best_err < tol_s:
        return None, leaf_type(data_set)
#    print best_feat, best_value
    right_set, left_set = bin_split(data_set, best_feat, best_value)
    if right_set.shape[0] < tol_n or left_set.shape[0] < tol_n:
        return None, leaf_type(data_set)
    return best_feat, best_value

# Note：如何设置终止条件是需要仔细考虑的问题！！
def create_tree(data_set, leaf_type = reg_leaf, err_type = reg_err, ops = (0.,4)):
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
#    print feat, val
    if feat == None:
        return val
    re_tree = {}
    re_tree["feature"] = feat
    re_tree["value"] = val
    right_set, left_set = bin_split(data_set, feat, val)
    re_tree["right"] = create_tree(right_set, leaf_type, err_type, ops)
    re_tree["left"] = create_tree(left_set, leaf_type, err_type, ops)
    return re_tree

def predict(re_tree, x):
    if type(re_tree) is not dict:
        return re_tree
    feat = re_tree["feature"]
    if x[feat] > re_tree["value"]:
        if type(re_tree["right"]) is dict:
            return predict(re_tree["right"], x)
        else:
            return re_tree["right"]
    else:
        if type(re_tree["left"]) is dict:
            return predict(re_tree["left"], x)
        else:
            return re_tree["left"]

            

if __name__ == "__main__":
#    np.random.seed(0)
#    test_data = np.random.random((200,10))
#    """
#array([[ 1.,  0.,  0.,  0.],
#       [ 0.,  1.,  0.,  0.],
#       [ 0.,  0.,  1.,  0.],
#       [ 0.,  0.,  0.,  1.]])
#    """
##    right_set, left_set = bin_split(test_data, 1, 0)
##    print "data1:\n",right_set
##    print "data2:\n",left_set
#    
#    feat, value = choose_best_split(test_data)
#    print "Best Feature:", feat, "\nBest Value:", value
#    
#    right_set, left_set = bin_split(test_data, feat, value)
#    print "Right Set:\n",right_set
#    print "LeftSet:\n",left_set
#    
#    re_tree = create_tree(test_data)

    data_set = load_dataset('ex0.txt')
    re_tree = create_tree(data_set,ops=(0.001,3))
    
#    pred = predict(re_tree, data_set[0,0:-1])
    
    pred = np.zeros((len(data_set),1))
    for i in xrange(len(data_set)):
        pred[i,:] = predict(re_tree, data_set[i,0:-1])
        
    import matplotlib.pyplot as plt
    plt.scatter(data_set[:,1],data_set[:,2],c='b',label = "Origin")
    plt.scatter(data_set[:,1],pred,c='r',label = "Fit")
    plt.legend()
    plt.show()
    
    pred = pred[data_set[:,1].argsort(),:]
    data_set = data_set[data_set[:,1].argsort(),:]
    plt.plot(data_set[:,1],data_set[:,2],'b-',label = "Origin")
    plt.plot(data_set[:,1],pred,'r-.',label = "Fit")
    plt.legend()
    plt.show()






























