# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 17:16:40 2017

@author: Young
"""

def loadDataSet():
    return [[1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]

def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # Note: 再Python中，frozenset是可哈希的，set是不可哈希的
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    """
    D：各条交易数据转换成的集合D = map(set, dataSet)
    Ck: 备选项集
    minSupport: 最小支持度
    """
    ssCnt = {}
    # 统计各个项集在数据集D中出现的次数
    for tid in D:
        for can in Ck:
            if can.issubset(tid):   # 如果项集是该交易的子集
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 对每个相集计算支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = sorted(list(Lk[i]))[0:k-2] # L1[0:0] = []
            L2 = sorted(list(Lk[j]))[0:k-2]
            if L1 == L2:
                # 在此处可以添加prune操作(在每产生一个候选k项集时，进行判断剪枝)
                ck = (Lk[i] | Lk[j])    # 产生候选
                if has_infrequent_subset(ck, Lk, k):
                    continue
                else:
                    retList.append(ck)
    """
    # 在此处可添加prune操作, 当k>2时（在所有候选k项集产生后，一并判断剪枝
    if k > 2:
        retList = pruneCk(retList, Lk, k)
    """
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)          # 产生候选频繁项集
        Lk, supK = scanD(D, Ck, minSupport) # 扫描数据库，根据最小支持度过滤项集
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def pruneCk(retList, Lk, k):
    '''
    Apriori：如果候选k项集的k-1项子集有一个不在Lk-1中，则该候选是不频繁的，
    应从生成的结果中滤除.
    '''
    retListPruned = []
    for ret in retList:
        # 对ret产生其所有的大小为k-1的子集，并判断其是否属于Lk
        # 算法：删除任意一个元素即可
        flag = True
        listRet = list(ret)
        for i in range(k):
            listTmp = listRet[:]    # 缓存拷贝，因为listTmp.pop()就地操作
            listTmp.pop(i)
            print listTmp
            #subRet.append(listTmp)
            
            # 判断ret子集是否属于Lk，如果存在一个不属于，则删去该项集ret
            if frozenset(listTmp) not in Lk:
                flag = False
                break
        if flag:
            retListPruned.append(ret)
    return retListPruned

def has_infrequent_subset(ck, Lk_1, k):
    ck_list = list(ck)
    #print ck
    #print Lk_1
    for i in range(k):
        ck_tmp = ck_list[:]
        ck_tmp.pop(i)
        #print ck_tmp
        if frozenset(ck_tmp) not in Lk_1:
            return True
    return False

if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = map(set, dataSet)
             