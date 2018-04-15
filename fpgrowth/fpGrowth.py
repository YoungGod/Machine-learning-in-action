# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:12:17 2017

@author: Young
"""

class treeNode:
    def __init__(self, name, numOccur, parentNode):
        self.name = name
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    
    def inc(self, numOccur):
        self.count += numOccur
    
    def disp(self, ind = 1):
        """
        深度优先disp树的结构
        """
        print '  '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项，创建满足最小支持度的1项集
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #print "满足最小支持度的1项集表：\n",headerTable
    # 若没有项满足，退出
    if len(freqItemSet) == 0:
        return None, None   
    # 构建头节点表，字典headerTable = {item：[count, treeNode], ...}     
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
        
    # 创建FP树
    retTree = treeNode('Null Set', 1, None)
    # 先根据全局频率对每个事务中元素排序，再插入retTree中
    for tranSet, count in dataSet.items():    # count表示事务计数，如果是非重复事务count=1
        localD = {}
        #print "原事务：", list(tranSet)
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
            if len(localD) > 0:
                orderedItems = [v[0] for v in sorted(localD.items(),
                                                     key = lambda p: p[1],
                                                     reverse = True)]
        #print "过滤排序后：", orderedItems
        # 将过滤并排序后的事务元素插入FP树中
        updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable
    
def updateTree(items, inTree, headerTable, count):
    # 将事务中元素依次从第一个开始递归插入inTree中，直到len(items) = 0
    # 如果已存在子树的孩子节点中，共享该元素，计数增1即count
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    # 如果不在孩子节点中，生成一个新的子节点，并链接在子树的根上
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头结点表的链接节点treeNode，初始为None
        # 即单链表尾插法updateHeader(..)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], 
                         inTree.children[items[0]])
    # 对剩下的事务元素递归插入子树tree中，每次将第一个元素插入
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]],
                   headerTable, count)
        
def updateHeader(nodeToTest, targetNode):
    # 单链表尾插法
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

'''
Test dataSet
'''
def loadSimpDat():
    simpDat = [['r','z','h','j','p'],
               ['z','y','x','w','v','u','t','s'],
               ['z'],
               ['r','x','n','o','s'],
               ['y','r','x','z','q','t','p'],
               ['y','z','x','e','q','s','t','m']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

"""
频繁模式挖掘，挖掘FP树
"""
def ascendTree(leafNode, prefixPath):
    # 递归上溯整棵树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    """
    # or 迭代
    while leafNode.parent != None:
        prefixPath.append(leafNode.name)
        leafNode = leafNode.parent
    """

def findPrefixPath(basePat, treeNode):
    # 通过头结点表的链接节点treeNode，找到关于basePat的路径集合构建其条件模式基
    condPats = {}        # 存储条件模式基（这里为前缀路径集合，及每条路径计数）
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 递归查找频繁项集
    # 从最不频繁的1项集开始，构建其条件模式基，条件树，最终构建与其相关的频繁模式
    bigL = [v[0] for v in sorted(headerTable.items(),
                                    key = lambda p: p[1])]
    for basePat in bigL:
        newFreqSet = preFix.copy()      # preFix类型为set，初始为空
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        # 构建条件模式基
        condPatBases = findPrefixPath(basePat, headerTable[basePat][1])
        # 构建条件树
        myCondTree, myHead = createTree(condPatBases, minSup)
        
        # 递归挖掘条件FP树，直到条件树中没有元素存在为止
        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        