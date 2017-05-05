# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 11:39:02 2017

@author: Young
"""
from math import log

def caculate_info_entropy(dataset):
    """
    Function: obtain the information entropy of a specific dataset
    ent = sum(-prob*log(prob))
    prob is the stochastic variable Y's(class number) distributon over the dataset.
    """
    ent = 0.0
    num_items = len(dataset)
    label_list = [item[-1] for item in dataset]
    label_counts = {label: label_list.count(label) for label in set(label_list)}
#    label_counts = {}
#    for label in label_list:
#        if label not in label_counts.keys():
#            label_counts[label] = 0
#        label_counts[label] += 1
    for key in label_counts:
        prob = float(label_counts[key])/num_items
        ent -= prob*log(prob,2)
    return ent

def split_dataset(dataset, atrr, value):
    """
    根据数据集中属性atrr(属性位置）的取值value划分得到新的数据数据集（即去掉该属性该取值后的数据集）
    """
    sub_dataset = []
#    sub_item = []
#    for item in dataset:
#        if item[atrr] == value:
#            sub_item.extend(item[:atrr])
#            sub_item.extend(item[atrr+1:])
#            sub_dataset.append(sub_item)
#            sub_item = []
    for item in dataset:
        if item[atrr] == value:
            sub_item = item[:atrr]
            sub_item.extend(item[atrr+1:])
            sub_dataset.append(sub_item)
    return sub_dataset

def choose_best_feature(dataset):
    """
    根据信息增益来选择最佳特征
    过程：循环根据每一个特征计算其信息增益并同时进行比较，总是记录最大的信息增益！
    思想转变：向量计算思想转换为循环，判断思想
    两种不同的解决问题的模式对应于两类问题：
    1. 非数值计算型程序设计问题
    2. 数值计算型程序设计问题
    
    for atrr = a1,a2..
        for a1 = a1_v1,a1_2..
            sub_dataset = split_dataset(a1,a1_v1)
            info_ent = info_ent(sub_dataset)
            info_ent = info_ent + len(sub_dataset)/len(dataset)*info_ent
        gain = info_ent(dataset) - info_ent
        if gain > largest_gain
            largest_gain = gain
            best_feature = atrr
    return best_feature, largetst,gain
    """
#    num_items = len(dataset)
#    dataset_ent = info_ent(dataset)
#    feature_list =  range(0,len(dataset[0])-1)
#    largest_gain = 0.0
#    best_feature = -1
#
#    for feature in feature_list:
#        ent = 0.0
#        feature_value_set = set([item[feature] for item in dataset])
#        for value in feature_value_set:
#            sub_dataset = split_dataset(dataset, feature, value)
#            sub_ent = info_ent(sub_dataset)
#            ent += float(len(sub_dataset))/num_items*sub_ent
#        gain = dataset_ent - ent
#        if gain > largest_gain:
#            largest_gain = gain
#            best_feature = feature
#    return best_feature
    num_samples = float(len(dataset))
    num_features = len(dataset[0]) - 1
    base_entropy = caculate_info_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    for feature in range(num_features):
        feature_value_set = set([sample[feature] for sample in dataset])
        new_entropy = 0.0
        for value in feature_value_set:
            sub_dataset = split_dataset(dataset, feature, value)
            prob = len(sub_dataset) / num_samples
            new_entropy += prob * caculate_info_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature
    
def majority_counts(class_label_list):
    """
    分别统计各类别标签数量，并返回数量最多的类别标签
    即：先分组计数，然后根据技术排序
    分组计数可以采用字典，集合来实现
    排序：对于字典排序，一般只能对键值排序
    方法，可以通过交换键值对
    """
#    class_counts = {label: class_label_list.count(label) for label in set(class_label_list)} # 计数
#    class_counts = {counts: label for label, counts in class_counts.items()} # 交换键值对
#    # 新构建的字典一般为字典序，对于数字来说即从小到大，故最大键值排在最后
#    class_counts = sorted(class_counts.items(), reverse=True)
#    return class_counts[0][1]
    import operator
    class_counts = {}
    for label in class_label_list:
#        if label not in class_counts.keys():
#            class_counts[label] = 0
        class_counts.setdefault(label,0)
        class_counts[label] += 1
    sorted_class_counts_list = sorted(class_counts.iteritems(),
                                      key = operator.itemgetter(1), # 1表示元组列表根据第二项来排序
                                      reverse = True)
    return sorted_class_counts_list[0][0]
 # items() 获得的是一个元组列表 
# iteritems()获得的是一个迭代器元组
# 如range() 相对于xrange()                                   
    
    
def create_decision_tree(dataset, labels):
    """
    首先，了解树的基本结构：
    节点类型：判断节点（记录特征），结论节点（记录类别）
    分支：记录特征取值
               根节点node                                  dataset
              /       \                      --->       /         \
    分支节点node1    分支节点node2      ...      sub_dataset1      sub_dataset2 ...
    a. 根据最佳特征进行分支，且每个特征只使用一次，因此树的深度为特征数量+1
    b. 分支数量等于特征的不同取值数，某节点的分支数量为当前特征取值数量
    c. 树的分支操作，实际上是根据特征的不同取值对原始数据集的划分
    
    结论节点：
    d. 当分支节点的子数据集中，特征为空时，即不可再划分，此时该节点标记为数据集中类被最多的那个类
    e. 当分支节点中子数据集中类别一致时，此时节点标记为该类
    
    判断节点：
    f. 当分支节点子数据集中，特征不为空且类别不一致时，此时节点为判断节点
       此时，在此节点重复分支划分过程直到各分支节点都为结论节点为止
       
    注意：树的基本结构是什么？ 根节点+分支节点（叶节点、新的根节点），然后再新的根节点上结构重复
    决策树需要存储的是什么？ 判断，分支（根据特征取值分支，特征有多少个取值就有多少个分支），结论（分支节点）
    
    嵌套字典：{判断：{特征取值1：结论1，特征取值2：结论2，特征取值3：{判断：...}}}
    基本结构：{判断：{特征取值：结论}}
    if 结论不为{}：则下结论（创建叶节点）
    否则：做判断（创建基本结构，重复操作）
    
    """
    class_label_list = [sample[-1] for sample in dataset]
    num_feature = len(dataset[0]) - 1
    
    # 终止条件，即创建结论节点，即节点的类别标签
#    if class_label_list.count(class_label_list[0]) == len(class_label_list):
    if len(set(class_label_list)) == 1:
        return class_label_list[0]
    if num_feature == 0:
        return majority_counts(class_label_list)
        
    best_feature_idx = choose_best_feature(dataset)
    best_feature_label = labels[best_feature_idx]
    tree = {best_feature_label:{}}
    del labels[best_feature_idx]
#    labels.remove(best_feature_label)

    best_feature_set = set([sample[best_feature_idx] for sample in dataset])
    
    for feature_value in best_feature_set:
        sub_dataset = split_dataset(dataset, best_feature_idx, feature_value)
        sub_labels = labels[:]    # 对列表修改的操作要特别注意，其是变化的！
        tree[best_feature_label][feature_value] = create_decision_tree(
                                                                 sub_dataset,
                                                                 sub_labels)
    return tree

    
#    class_label_list = [sample[-1] for sample in dataset]
#    feature_num = len(dataset)-1
#    if len(set(class_label_list)) == 1:
#        return class_label_list[0]
#    if feature_num == 0:
#        return majority_counts(class_label_list)
#        
#    best_feature = choose_best_feature(dataset)
#    best_feature_value_set = set([sample[best_feature] for sample in dataset])
#    best_label = labels[best_feature]
#    del labels[best_feature]
#    
#    tree = {best_label:{}}
#    for value in best_feature_value_set:
#        sub_dataset = split_dataset(dataset, best_feature, value)
#        tree[best_label][value] = create_decision_tree(sub_dataset, labels)
#    return tree

def get_num_leaf(tree):
    """
    从嵌套字典总获取叶节点数量
    例如：
    tree = {'No Surfacing?': {0: 'no', 1: {'Flippers?': {0: 'no', 1: 'yes'}}}}
    我们把一个根节点及其分支节点看成一个树的基本结构，见函数create_decision_tree()
    其他延伸只不过是该结构的重复！
    那么，先考虑统计基本结构中的叶节点：
    例如：
    把tree看成 {'No Surfacing?': {0: 'no', 1: '重复结构'}}
    '重复结构' = {'Flippers?': {0: 'no', 1: 'yes'}}
    即根节点+分支
    其中键'Flippers'为根节点，值为{0: 'no', 1: 'yes'}，其构成2分支，组成两个键值对
    """
#    leaf_num = 0
#    root_node = tree.keys()[0]
#    for key in tree[root_node]:
#        if type(tree[root_node][key]) is dict:
#            leaf_num += get_num_leaf(tree[root_node][key])
#        else:
#            leaf_num += 1
#    return leaf_num
    num_leaf = 0
    first_str = tree.keys()[0]  # 即根节点（第一个字典存储根节点）
    second_dict = tree[first_str] # 即分支节点（第二个字典存储分支情况，其嵌套在第一个字典中
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leaf += get_num_leaf(second_dict[key])
        else:
            num_leaf += 1
    return num_leaf

def get_tree_depth(tree):
    """
    树的深度即树的层数.
    举例：
    tree = {'No Surfacing?': {0: 'no', 1: {'Flippers?': {0: 'no', 1: 'yes'}}}}
    树的深度即嵌套字典数量
    而在机器学习实战一书中，深度定义为判断节点数，即节点层数-1
    """
#    max_depth = 0
    first_str = tree.keys()[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
    return this_depth
           
def predict(tree, labels, sample_feature):
    """
    预测流程：根据根据特征以及特征取值判定其所属叶子节点（即类别）
    例如：sample_feature = [0, 1]
    tree = {'No Surfacing?': {0: 'no', 1: {'Flippers?': {0: 'no', 1: 'yes'}}}}
    如果特征的取值（在树中表为嵌套字典的键值）能够达到叶子节点（其值为非字典类型），
    则返回该键的取值，即类别标签
    否则，递归！
    一定要清楚树在字典中是如何存储的！！
    
    树的存储-嵌套字典，及其存储的内容：
    {
     特征判断：
            {
               特征取值：类别标签， 特征取值：{特征判断：..}
            }
    }
            
    """
    root_node = tree.keys()[0]
    idx = labels.index(root_node)
#    print "index:", idx
    second_dict = tree[root_node]
    for key in second_dict.keys():
#        if sample_feature[idx] == key:
#            if type(second_dict[key]) is dict:
#                class_label = predict(second_dict[key], labels, sample_feature)
#            else:
#                class_label = second_dict[key]
#    return class_label
        if sample_feature[idx] == key:
            if type(second_dict[key]) is dict:
                return predict(second_dict[key], labels, sample_feature)
            else:
                return second_dict[key]
   
def store_tree(tree, filename):
    """
    功能：把训练好的树存储到磁盘
    """
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()
    
def grab_tree(filename):
    """
    功能：从磁盘取出训练好的树
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)

## test
#dataset = [[1,1,'yes'],
#           [1,1,'yes'],
#           [1,0,'no'],
#           [0,1,'no'],
#           [0,1,'no']]
#dataset1 = [[1,1,1,'yes'],
#           [1,1,0,'yes'],
#           [1,0,1,'no'],
#           [0,1,0,'no'],
#           [1,0,0,'no']]
##print info_ent(dataset)
##print split_dataset(dataset, 0, 1)
##print choose_best_feature(dataset)
#tree = create_decision_tree(dataset,labels = ["No Surfacing?","Flippers?"])
#store_tree(tree, 'TREE.txt')
#tree = grab_tree('TREE.txt')
#print "Tree:", tree
#print "Number of leaf node:", get_num_leaf(tree)
#print "Depth:", get_tree_depth(tree)
#print "Prediction:", predict(tree, labels = ["No Surfacing?","Flippers?"],
#                             sample_feature = [0,1])

# application
"""
隐形眼镜配戴类型预测,即根据患者特征及眼部观察条件（特征），以及医生推荐的眼镜（类别）
向患者推荐合适的眼镜
"""
fr = open("lenses.txt")

#dataset = fr.read()
#dataset = [sample.split('\t') for sample in dataset.strip().split('\n')]
dataset = [sample.strip().split('\t') for sample in fr.readlines()]
fr.close()
labels = ["age","prescript","astigmatic","tearRate"]

# train
lenses_tree = create_decision_tree(dataset[:-4], labels)

# prediction
test_data_set = dataset[-4:]
test_set_feature = []
test_class_label = [sample[-1] for sample in test_data_set]
for sample in test_data_set:
    test_set_feature.append(sample[:-1])
    
labels = ["age","prescript","astigmatic","tearRate"] 
#predict(lenses_tree, labels, test_set_feature[0])

predict_lenses = []
for sample_feature in test_set_feature:
    predict_lenses.append(predict(lenses_tree, labels, sample_feature))

print "Origin:",test_class_label,"\n","Predict:",predict_lenses

# note: 程序测试与函数编写需要分开，因为函数中的变量会影响后面的变量
# 当训练数据不足时，对于有些样本，决策树不能给出其分类，这个问题需要debug