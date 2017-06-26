# -*- coding: utf-8 -*-
"""
Created on Mon May 01 13:30:53 2017

@author: Young
"""

"""
1 贝叶斯分类原理
特征空间及输出空间：（X，y），X = [x1,x2,...xn], y∈{0,1,..,k} (n个特征，k个类别)

分类即计算及比较条件概率：给定特征输入X，判断其属于y = i，i=0,1,..,k类的概率，然后
                        比较各条件概率大小，概率最大的那个即输入X所属的类别
对于第i类条件概率 P(y=i|X) = P(X,y=i) / P(X)
                = P(X|y)P(y) / P(X)
其中未知参数为：P(X|y), P(y)，P(X)

那么，学习的任务就是学习参数：P(X|y), P(y), P(X)
a. 由于对于每一类都要计算条件概率，而只需最后比较条件大小。观察到分母P(X)对于每个
   概率都是一样的，在只比较大小时，其无影响，可以约去。因此，对于该参数可以不学习！
b. 条件概率P(X|y=i) = P(x1,x2,...,xn|y=i),由于X是n维随机变量，求P(x1,x2,...,xn|y=i)
   即在类别y=i下，求(x1,x2,...,xn)的n维联合概率分布，若每个随机变量取值个数为100，则
   要学习的参数数量为100*100*..*100 = 100^n,参数数量庞大，为防止过拟合，则需要大量
   训练数据
   若假设各个特征条件独立，即在给定类别y=i下，xi与xj没有关系，即xi的出现不会影响xj
   出现的概率，反之亦然。
   举例：文本分类中，特征 = {“苹果”，“不好吃”，“好吃”},我们知道“苹果”这个特征出现在
   “好吃”前面的概率应该要比出现在“不好吃”前面的概率大，因此“苹果”这个特征的出现影响
   了特征“不好吃”，尤其是“好吃”出现的概率 （好像不太合适）
   继续，假设特征条件独立，则联合概率：
   P(x1,x2,...,xn|y=i) = P(x1|y=i)*P(x2|y=i)*...*P(xn|y=i)
   此时，学习的参数为各个随机变量在给定类别y=i下的n个一维随机分布   
   此时，若每个随机变量取值为100，则需要学习的参数为100+100+...+100 = 100*N
   可见，参数大大减少！
c. 特征条件独立的假设虽然是个强假设，但据说实际效果还不错。且因为只需要分别学习各个
   随机变量的条件概率分布，一方面，使得学习参数大幅减少，减少了对数据量的要求；另一
   方面，对于特征X中的连续型变量及连续型随机变量也能处理，根据数据估计其连续分布即可
   
d. 贝叶斯具体实现方式。查表法，实时计算法，增量学习法（见机器学习周志华）
   查表法在python实现可以采用嵌套字典来实现！采用键值对方式实现！！

e. 注意问题：
   在给定一个新输入X时，分别计算器所属第i类条件概率
   P(y=i|X) = P(X,y=i) / P(X) = P(X|y)P(y) / P(X) (P(X)可忽略,其对结果判断无影响)
   然后比较各条件概率大小，概率最大的那个即输入X所属的类别.
   
   但是，计算其所属类别的条件概率时：
   因为P(X|yi=i) = P(x1,x2,...,xn|y=i) = P(x1|y=i)*P(x2|y=i)*...*P(xn|y=i)
   在所属类别i中，某特征变量xi的某个取值没有出现，即xi有多个取值，但有些取值没有在
   类别i中出现，此时会得到P(xi |y = j) = 0的情况，则P(y=i|X) = 0.
   此时可能由于某个别特征变量取值没出现而造成误判。
   
   为避免这样的情况发生，可采用平滑法，也称贝叶斯估计法。
   如预先确保所有特征变量xi的取值都在类别j中出现一次，这种方法为laplace平滑法。
   
   参数估计时，注意调整，使得平滑后的分布满足概率分布条件！（见统计学习方法李航）


2 贝叶斯文本分类流程
   a. 根据现有文档构建文档所有不重复词汇列表向量
   b. 根据词汇列表向量，构建文档特征向量(采用伯努利计数法或多项式计数法)
   c. 贝叶斯估计(对于文本的分类处理，可能与标准的估计不一致，机器学习实战中与统计学习基础中不一样！)
"""

def load_dataset():
    """
    列表中每一项表示一个文档，文档由单词构成，长度不一；
    对于英文来说分词较为容易，采用了空格；
    对于中文来说需要采用分词程序，构成有一个个单词构成的文档.
    """
    posting_list = [
            ['my', 'dog', 'has', 'flea', 'problems','help', 'please'],
            ['maybe', 'not', 'take', 'him','to','dog','park','stupid'],
            ['my','dalmation','is','so','cute','I','love','him'],
            ['stop','posting','stupid','worthless','garbage'],
            ['mr','licks','ate','my','steak','how','to','stop','him'],
            ['quit','buying','worthless','dog','food','stupid']
            ]
    class_label_list = [0, 1, 0, 1, 0, 1]    # 1代表侮辱性文字，0代表正常
    return posting_list, class_label_list

def create_voca_list(dataset):
    """
    功能：创建词汇列表
    词汇列表：对于某一个分类预测问题来说，文档中所有可能出现的词汇集合（不重复）
    词汇列表构成了文档的所有特征，一个词汇即一个特征！
    因此，构建词汇列表及构建学习问题的特征空间.词汇列表的长度及特征属性的数量，特征空间的维度
    所以，构建词汇列表是进行特征选择.
    一般来说，对于一般文档分类问题，词汇列表应该包含几乎可能出现的所有词汇，这样特征空间
    维度非常大.
    方式：1. 通过遍历所有文档，提取所有可能词汇 2. 采用预先给定的通用词汇列表
    这里采用遍历文档，来构建词汇列表
    """
    voca_list = set()
    for document in dataset:
        voca_list = voca_list | set(document)
    return list(voca_list)
    
##    voca_list = []
##    for document in dataset:
##        for word in document:
##            voca_list.append(word)
##    return list(set(voca_list))

#    voca_set = set()
#    for document in dataset:
#        voca_set = voca_set | set(document)
#    return list(voca_set)

def caculate_words_vector(voca_list, document):
    """
    根据词汇列表（特征属性），给定一个文档数据，构建该文档的特征属性值
    在python中使用list这个可变数据结构时要特别小心！！
    """
    words_vector = [0]*len(voca_list)
    for word in document:
        if word in voca_list:
            idx = voca_list.index(word)
            words_vector[idx] += 1
        else:
            print "The word is not in my vocabulary!"
    return words_vector
    
##    # 注释的程序试图处理数据dataset中，所有document的向量化，存在每次都要初始化list问题
##    # 容易出错
##    words_vector_list = []
##    words_vector = [0]*len(voca_list)    # 初始化单个文档的特征属性取值
##    for document in dataset:
##        for word in document:
##            if word in voca_list:
##                idx = voca_list.index(word)
##                words_vector[idx] = 1    # 在重复使用list时要非常小心！其是可变的！！
##            else:
##                print "The word %s is not in vocabulary." % word
##        words_vector_list.append(words_vector)
##        words_vector = [0]*len(voca_list)    # 重新初始化list
##    return words_vector_list
#
#    # 一次处理一个文档
#    words_vector = [0]*len(voca_list)
#    for word in document:
#        if word in voca_list:
#            words_vector[voca_list.index(word)] = 1
#        else:
#            print " The word %s is not in vocabulary." % word
#    return words_vector

import numpy as np
def train_bayes0(words_vector_list, class_label_list):
    """
    功能：求先验概率和条件概率
    注意：二分类情况，且特征取值0或1
    一个numpy中的array + list = array，及list类型自动变为array！
    """
    num_class1 = sum(class_label_list)
#    label_counts = {}
#    for label in class_label_list:
#        label_counts.setdefault(label,0)
#        label_counts[label] += 1
    num_vectors = len(words_vector_list)
    prob_class1 = num_class1 / float(num_vectors)
    prob_class0 = 1 - prob_class1
    
    num_words_class1 = np.ones(len(words_vector_list[0]))
    num_words_class0 = np.ones(len(words_vector_list[0]))
    
    num_total_words_class1 = 2.0
    num_total_words_class0 = 2.0
    
    for i in xrange(num_vectors):
        if class_label_list[i] == 1:
            num_words_class1 += words_vector_list[i]
            num_total_words_class1 += sum(words_vector_list[i])
        else:
#            print
            num_words_class0 += words_vector_list[i]
            num_total_words_class0 += sum(words_vector_list[i])
    prob_words_class1 = np.log(num_words_class1 / num_total_words_class1)
    prob_words_class0 = np.log(num_words_class0 / num_total_words_class0)
    
    return prob_words_class1, prob_words_class0, np.log(prob_class1), np.log(prob_class0)
   
#    num_vectors = len(words_vector_list)
#    num_words = len(words_vector_list[0])
#    
#    prob_class1 = sum(class_label_list) / float(num_vectors)
#    prob_class0 = 1 - prob_class1
#    
#    p0_num = np.zeros(num_words)
#    p1_num = np.zeros(num_words)
##    print p0_num
#    p0_denom = 0.0
#    p1_denom = 0.0
#    
#    for i in range(num_vectors):
#        if class_label_list[i] == 1:
#            p1_num += words_vector_list[i]
#            p1_denom += sum(words_vector_list[i])
#        else:
#            p0_num += words_vector_list[i]
#            p0_denom += sum(words_vector_list[i])
#            
#    prob1_vector = p1_num / p1_denom
#    prob0_vector = p0_num / p0_denom
#    
#    return prob0_vector, prob1_vector, prob_class1, prob_class0

       
def classify_NB(words_vector, prob_words_class1,
                prob_words_class0, prob_class1, prob_class0):
    prob1 = sum(words_vector*prob_words_class1) + prob_class1
    prob0 = sum(words_vector*prob_words_class0) + prob_class0
    
    if prob1 > prob0:
        return 1
    else:
        return 0

def testing_NB():
    dataset, class_label_list = load_dataset()
#    print dataset
    # 根据所有文档，构建词汇表
    voca_list = create_voca_list(dataset)
    # 构建文档词汇向量，或者说文档的向量化（set of words）
    words_vector_list = []
    for document in dataset:
        words_vector_list.append(caculate_words_vector(voca_list, document))
    # 贝叶斯学习，求解各特征即词汇的条件概率分布，以及类别的先验分布
    prob1_vector, prob0_vector, prob_class1, prob_class0 =\
            train_bayes0(words_vector_list, class_label_list)
    # 引入测试文档，进行预测
    test_document = ["love", "my", "dalmation", "garbage"]
    test_words_vector = caculate_words_vector(voca_list, test_document)
    print test_document,\
    " classified as:",\
    classify_NB(test_words_vector,prob1_vector,prob0_vector,prob_class1,prob_class0)
    
    test_document = ["stupid", "garbage", "love"]
    test_words_vector = caculate_words_vector(voca_list, test_document)
    print test_document,\
    " classified as:",\
    classify_NB(test_words_vector,prob1_vector,prob0_vector,prob_class1,prob_class0)

# test
#testing_NB()


## test
#dataset, class_label_list = load_dataset()
#voca_list = create_voca_list(dataset)
#
#words_vector_list = []
#for document in dataset:
#    words_vector_list.append(caculate_words_vector(voca_list, document))
#print words_vector_list
#
#prob1_vector, prob0_vector, prob_class1, prob_class0 =\
#            train_bayes0(words_vector_list, class_label_list)   
    
    