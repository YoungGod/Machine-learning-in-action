# -*- coding: utf-8 -*-
"""
Created on Fri May 05 11:38:02 2017

@author: Young
"""
import bayes
import re
#import string
## test
#bayes.testing_NB()

"""
使用贝叶斯对电子邮件分类流程：
1. 收集邮件文本
2. 将文本文件解析成词条向量，并确保正确性
3. 训练
4. 测试，使用

NOTE: coding a little, fix a little!!

"""
## for test
#spam_source = "email/spam/%s.txt"
#ham_source = "email/spam/%s.txt"
#
#spam_list = []
#ham_list = []
#for i in range(1,26):
#    spam_list.append(open(spam_source % i).read())
#    ham_list.append(open(ham_source % i).read())
#regex = re.compile('\\W*') # W与w相反
#words_list01 = regex.split(spam_list[0])
#words_list0 = [word.lower() for word in words_list01 
#                                if len(word) > 0 and word.isalpha()]

# text 解析！！
def text_parse(text):
#    import re
    words_list = re.split(r'\W*', text)
    return [word.lower() for word in words_list if len(word) > 2]

def spam_test():
    doc_list = []; class_list = []; full_text = []
    for i in range(1,26):
        words_list = text_parse(open('email/spam/%d.txt'%i).read())
        doc_list.append(words_list)    # 用于构建训练和测试集合
        full_text.extend(words_list)    # 用于构建文件字典
        class_list.append(1)
        
        words_list = text_parse(open('email/ham/%d.txt'%i).read())
        doc_list.append(words_list)
        full_text.extend(words_list)
        class_list.append(0)
    print "OK"    
    vocabulary = bayes.create_voca_list(doc_list)
    
    training_set = range(50); test_set = []
    for i in range(10):   # 不放回随机抽样构建测试集(由索引构成)
        rand_idx = int(np.random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_idx])
        del (training_set[rand_idx])
    
    train_vector_list = []; train_class_list = []
    for doc_idx in training_set:
        train_vector_list.append(bayes.caculate_words_vector(vocabulary,doc_list[doc_idx]))
        train_class_list.append(class_list[doc_idx])
        
    prob1, prob0, prob_spam, prob_ham = bayes.train_bayes0(train_vector_list,
                                                           train_class_list)
    error_count = 0
    for doc_idx in test_set:
        test_vector = bayes.caculate_words_vector(vocabulary,doc_list[doc_idx])
        if bayes.classify_NB(test_vector,prob1,prob0,prob_spam,prob_ham) != \
                            class_list[doc_idx]:
            error_count += 1
    print "The error rate is : ",float(error_count)/len(test_set)

spam_test()














