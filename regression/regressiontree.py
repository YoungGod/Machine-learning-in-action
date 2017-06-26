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
    print "err",err_type(data_set) - best_err
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

def is_tree(obj):
    return (type(obj).__name__ == 'dict')

def get_mean(tree):
     if is_tree(tree['right']):
         tree["right"] = get_mean(tree["right"])
     if is_tree(tree['left']):     
         tree["left"] = get_mean(tree["left"])
     return (tree["right"]+tree["left"])/2.0

def prune(tree, test_data):
    if test_data.shape[0] == 0:
        return get_mean(tree)
    if is_tree(tree["right"]) or is_tree(tree["left"]):
        right_set, left_set = bin_split(test_data, tree["feature"], tree["value"])
    if is_tree(tree["right"]):
        tree["right"] = prune(tree["right"], right_set)
    if is_tree(tree["left"]):
        tree["left"] = prune(tree["left"], left_set)
    if not is_tree(tree["right"]) and not is_tree(tree["left"]):
        right_set, left_set = bin_split(test_data, tree["feature"], tree["value"])
        err_no_merge = sum(np.power(right_set[:,-1] - tree["right"], 2)) + \
                          sum(np.power(left_set[:,-1] - tree["left"], 2))  # 预测误差平方和
        tree_mean = (tree["right"] + tree["left"])/2.0
        err_merge = sum(np.power(test_data[:,-1] - tree_mean, 2))
        if err_merge < err_no_merge:
            print "merging"
            return tree_mean
        else:
            return tree
    else:
        return tree

def get_num_leaf(tree):
    
    if not is_tree(tree):
        return 1
    else:
        return get_num_leaf(tree["right"])+get_num_leaf(tree["left"])

def get_num_node(tree):
    num_node = 0
    if not is_tree(tree):
        return 0
    if is_tree(tree):
        num_node = 1 + get_num_node(tree["right"]) + get_num_node(tree["left"])
    return num_node

        

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
    
    # Fitting
    data_set = load_dataset('ex0.txt')
    re_tree = create_tree(data_set,ops=(0.00001,2))
    
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

    # testing
    test_data = load_dataset('ex1.txt')
    pred = np.zeros((len(test_data),1))
    for i in xrange(len(test_data)):
        pred[i,:] = predict(re_tree, test_data[i,0:-1])    
    err_no_prune = sum(np.power(test_data[:,-1] - pred.flatten(),2))
    
    pred = pred[test_data[:,1].argsort(),:]
    test_data = test_data[test_data[:,1].argsort(),:]
    plt.plot(test_data[:,1],test_data[:,2],'b-',label = "Origin")
    plt.plot(test_data[:,1],pred,'r-.',label = "Predict")
    plt.legend()
    plt.show()    
    
    # pruning
    test_data = load_dataset('ex1.txt')
    pruned_tree = prune(re_tree, test_data)

    pred = np.zeros((len(data_set),1))
    for i in xrange(len(data_set)):
        pred[i,:] = predict(pruned_tree, data_set[i,0:-1])    

    plt.scatter(data_set[:,1],data_set[:,2],c='b',label = "Origin")
    plt.scatter(data_set[:,1],pred,c='r',label = "Pruned Fit")
    plt.legend()
    plt.show()
    
    pred = pred[data_set[:,1].argsort(),:]
    data_set = data_set[data_set[:,1].argsort(),:]
    plt.plot(data_set[:,1],data_set[:,2],'b-',label = "Origin")
    plt.plot(data_set[:,1],pred,'r-.',label = "Pruned Fit")
    plt.legend()
    plt.show()
    
    # testiing
    pred = np.zeros((len(test_data),1))
    for i in xrange(len(test_data)):
        pred[i,:] = predict(pruned_tree, test_data[i,0:-1])    

    pred = pred[test_data[:,1].argsort(),:]
    test_data = test_data[test_data[:,1].argsort(),:]
    plt.plot(test_data[:,1],test_data[:,2],'b-',label = "Origin")
    plt.plot(test_data[:,1],pred,'r-.',label = "Predict")
    plt.legend()
    plt.show()  
    err_prune = sum(np.power(test_data[:,-1] - pred.flatten(),2))
    
    print "Error:", err_no_prune
    print "Error Pruned:", err_prune
    
    
    tree = {"right":{"right":{"right":5, "left":6}, "left": 4}, "left":2}
    


    import copy
    data_set = load_dataset('ex0.txt')
    re_tree = create_tree(data_set,ops=(0.0001,1))
      
    to_pruned_tree = copy.deepcopy(re_tree)
    test_data = load_dataset('ex1.txt')
    pruned_tree = prune(to_pruned_tree, test_data)
    
    print get_num_leaf(re_tree)
    print get_num_leaf(pruned_tree)
    
    print get_num_node(re_tree)
    print get_num_node(pruned_tree)
    
    print id(re_tree)
    print id(pruned_tree)



    
    np.random.seed(0)
    X = np.random.random((100,4))
    y = np.random.randint(0,10,100)
    dataset = np.concatenate((X,y.reshape(-1,1)),axis=1)
    re_tree = create_tree(dataset,ops=(0.0000,1))
    print re_tree





















