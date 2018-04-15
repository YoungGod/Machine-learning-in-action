# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 05:42:02 2017

@author: Young
"""

import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    dataset = np.array([
            [1., 2.1],
            [1.5, 1.6],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]])
    label = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return dataset, label

def plot_scatter(dataset,label):
    """
    scatter plot
    """
    fig, ax = plt.subplots()
    label_unique = np.unique(label)
    colors = ['r','b']
    labels = ['red','blue']
    for c,l,la in zip(colors,label_unique,labels):
        index = (label==l)
        ax.scatter(dataset[index,0],dataset[index,1],color=c,label=la)
    ax.legend()

def stump_classify(dataset, feat, thresh_val, thresh_ineq):
    prediction = np.ones(dataset.shape[0])
    if thresh_ineq == 'lt':
        prediction[dataset[:,feat] <= thresh_val] = -1.0
    else:
        prediction[dataset[:,feat] > thresh_val] = -1.0
    return prediction
    
def build_stump(dataset,label,weights):
    m_instance, n_feat = dataset.shape
    num_steps = 10
    min_err = np.inf
    best_stump = {}; best_class_estimated = np.zeros((m_instance,1)) 
    for feat in xrange(n_feat):
        feat_min = dataset[:,feat].min(); feat_max = dataset[:,feat].max()
        step_size = (feat_max - feat_min) / num_steps
        for value in xrange(-1,num_steps+1):
            for inequal in ['lt','gt']:
                thresh_val = feat_min + float(value) * step_size
                predicted_vals = stump_classify(dataset, feat, thresh_val, inequal)
                err_arr = np.ones(m_instance)
                err_arr[predicted_vals == label] = 0  # 预测正确的err修改为0
                weighted_err = sum(weights * err_arr)
#                print "OKOKOKOKOKOKOO"
#                print "split: feature %d, thresh %.2f, thresh inequal:\
#                %s, the weighted error is %.3f" %\
#                (feat, thresh_val, inequal, weighted_err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_estimated = predicted_vals.copy()
                    best_stump['feat'] = feat
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_err, best_class_estimated   

def adaboost_train(dataset, label, num_iter=2):
    weak_classifiers = []
    m_instance = dataset.shape[0]
    weights = np.ones(m_instance)/m_instance
    agg_prediction = np.zeros(m_instance)
    err_rates = []
    for i in xrange(num_iter):
        agg_err = np.ones(m_instance)
        best_stump, err, prediction = build_stump(dataset, label, weights)
        alpha = 0.5*np.log((1-err)/max(err,1e-16))    # 防止err为0，除数为0
#        print err,alpha
        best_stump['alpha'] = alpha
        weak_classifiers.append(best_stump)
        expon = -alpha*label*prediction    # 二分类问题，向量化计算
        weights = weights*np.exp(expon)/weights.sum()
#        print weights
        agg_prediction += alpha*prediction  # 预测
#        print np.sign(agg_prediction)
        agg_err[np.sign(agg_prediction)==label] = 0
        err_rate = agg_err.mean()
#        print agg_err
#        print err_rate
        err_rates.append(err_rate)
        if err_rate == 0.0: break
    return weak_classifiers, np.array(err_rates)

def adaboost_predict(X, weak_classifiers):
    """
    X 为预测的样本(m_insatance, n_feat)
    """
    agg_prediction = np.zeros(len(X))
    for classifier in weak_classifiers:
        prediction = stump_classify(X, classifier['feat'],classifier['thresh'],classifier['ineq'])
#        print prediction
        agg_prediction += classifier['alpha']*prediction
    return np.sign(agg_prediction), agg_prediction

def load_data(filename):
    fr = open(filename)
    dataset = []; label = []
    for line in fr:
        instance = [float(x.strip()) for x in line.split('\t')]
        dataset.append(instance[0:-1])
        label.append(instance[-1])
    return np.array(dataset), np.array(label)

def plot_roc(pred_strengths, class_label):
    cur = (1.0, 1.0)
    y_sum = 0
    num_pos_class = sum(np.array(class_label)==1.0)
    y_step = 1/float(num_pos_class)
    x_step = 1/float(len(class_label)-num_pos_class)
    sorted_index = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_index:
        if class_label[index] == 1.0:
            delt_x = 0; delt_y = y_step;
        else:
            delt_x = x_step; delt_y = 0;
            y_sum += cur[1]
        ax.plot([cur[0],cur[0]-delt_x],[cur[1],cur[1]-delt_y],c='b')
        cur = (cur[0]-delt_x, cur[1]-delt_y)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for AdaBoost Horse Colic Detection System")
    ax.axis([0,1,0,1])
    plt.show()
    print "AUC is: ", y_sum*x_step

if __name__ == "__main__":
#    dataset, label = load_dataset()
#    plot_scatter(dataset,label)
#    weights = np.ones(5)/5
#    best_stump, min_err, best_class_estimated = build_stump(dataset,label,weights)
#    weak_classifiers, err_rates = adaboost_train(dataset, label, num_iter=9)
#    prediction = adaboost_predict(dataset, weak_classifiers)
    
    # horseColic train &  predict
    X, y = load_data('horseColicTraining2.txt')
    num_iter = 45
    weak_classifiers, err_rates = adaboost_train(X, y, num_iter=num_iter)
    fig, ax = plt.subplots()
    ax.plot(range(1,num_iter+1),err_rates,label='error')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('AggError')
    ax.legend()
    print "Training Error", err_rates[-1]
    
    X_test,y_test = load_data('horseColicTest2.txt')
    prediction, strength = adaboost_predict(X_test, weak_classifiers)
    err = np.ones(len(X_test))*(prediction!=y_test)
    err_rate = err.mean()
    print "Testing Error", err_rate
    plot_roc(strength,y_test)
    
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=1)
    clf.fit(X,y)
    prediction = clf.predict(X_test)
    err = np.ones(len(X_test))*(prediction!=y_test)
    err_rate = err.mean()
    print "Sklearn Testing Error", err_rate
    
   
    
    
    
    