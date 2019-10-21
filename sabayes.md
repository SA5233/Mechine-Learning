# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:43:58 2019

@author: 韩琳琳
"""

###################鸢尾花实验

import numpy as np
import pandas as pd
import random
import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch04')

dataset=pd.read_csv('iris.txt',header=None)


#切分训练级和测试集
def randsplit(dataset,rate):
    l=list(dataset.index)
    random.shuffle(l)
    dataset.index=l
    n=dataset.shape[0]
    m=int(n*0.8)
    train=dataset.loc[range(m),:] #注意不能写成.iloc or [:m,:]
    test=dataset.loc[range(m,n),:]
    test.index=range(n-m)
    dataset.index=range(n)
    return train,test
train,test=randsplit(dataset,0.8)

#构建贝叶斯分类器
def gauss_clsssify(train,test):
    labels=train.iloc[:,-1].value_counts().index 
    mean=[]
    std=[]
    for la in labels:
        item=train.loc[train.iloc[:,-1]==la] # loc 列名不能为-1
        m=item.iloc[:,:-1].mean()
        s=np.sum((item.iloc[:,:-1]-m)**2)/item.shape[0]
        mean.append(m)
        std.append(s)
    mean=pd.DataFrame(mean) #mean=pd.DataFrame(mean,index=labels)
                            #这样后面的 pla=p.index[np.argmax(p.values)]
    std=pd.DataFrame(std)
    result=[]
    
    for i in range(test.shape[0]):
        iest=test.iloc[i,:-1].tolist() #当前测试实例
        pr=np.exp(-(iest-mean)**2/(2*std))/np.sqrt(2*np.pi*std) #得到正态分布概率矩阵
        p=1
        for j in range(test.shape[1]-1):
            p*=pr[j]
            pla=labels[p.index[np.argmax(p.values)]]
        result.append(pla)
    retest = test.copy()                    #这样程序执行完毕后，test不改变
    retest['predict'] = result
    accuracy = (retest.iloc[:,-1]==retest.iloc[:,-2]).mean()
    print(f'模型的预测准确率为{accuracy}')
    

for i in range(20):
    train ,test=randsplit(dataset,0.8)
    gauss_clsssify(train,test)
    

############################## sklearn #########################################

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import datasets
iris=datasets.load_iris()

#切分数据集
Xtrain,Xtest,ytrain,ytest=train_test_split(iris.data,iris.target,random_state=42)#随机数种子决定不同切分规则
#建模
clf=GaussianNB()
clf.fit(Xtrain,ytrain)
#在测试集上执行预测，proba导出的是每个样本属于某一类的概率
clf.predict(Xtest)
clf.predict_proba(Xtest)
#测试准确率
accuracy_score(ytest,clf.predict(Xtest))

#连续性用高斯贝叶斯，0-1用伯努利贝叶斯，分词用多项式朴素贝叶斯




#################################朴素贝叶斯之言论过滤

import numpy as np

def loadDataSet():
    dataset=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 侮辱性词表, 0 非侮辱性词表
    return dataset,classVec

dataset,classVec = loadDataSet()
                 
#构建词汇表

def creatVocabList(dataset):
    vocablist=set()     #只有set和set才能取并集
    for doc in dataset:
        vocablist=vocablist|set(doc) #并集
    vocablist=list(vocablist) 
    #vocablist=set(vocablist)   #并集的结果已经去过重了
    return vocablist

vocablist = creatVocabList(dataset)

#获得训练集向量

def setOfWords2Vec(vocablist,inputset): #输入词表和切分好的一个词条
    returnVec = [0]*len(vocablist)    #与词表等长的零向量
    for word in inputset:
        if word in vocablist:
            returnVec[vocablist.index(word)] = returnVec[vocablist.index(word)]+1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def get_trainMat(dataset):
    vocablist=creatVocabList(dataset)
    result=[]
    for inputset in dataset:
        vec=setOfWords2Vec(vocablist,inputset)
        result.append(vec)
    return result

trainMat = get_trainMat(dataset)

#朴素贝叶斯分类器训练函数

def trainNB(trainMat,classVec):
    n = len(trainMat)   #总文档数目
    m = len(trainMat[0]) #所有文档中非重复词条数
    pA0 = sum(classVec)/n  #侮辱性文档占总文档的概率
    p0num = np.zeros(m)   # 初始化
    p1num = np.zeros(m)
    p1demo = 0
    p0demo = 0
    for i in range(n):
        if classVec[i]==1:       
            p1num += trainMat[i] # 侮辱性文档中词条的分布
            p1demo += sum(trainMat[i]) #侮辱性文档中词条总数
        else:
            p0num += trainMat[i]
            p0demo += sum(trainMat[i])
    p1v = p1num/p1demo       #全部侮辱类词条的条件概率数组
    p0v = p0num/p0demo 
    return p1v,p0v,pA0 

p1v,p0v,pA0 = trainNB(trainMat,classVec)

#测试朴素贝叶斯分类器
 
from functools import reduce

def classifyNB(vec2classify,p1v,p0v,pA0): # vec2classify 待分类的词条分布数组
    p1 = reduce(lambda x,y:x*y,vec2classify*p1v)*pA0  #reduce作用，对应数字相乘(已知词组属于侮辱类的条件概率*pA0 )
    p0 = reduce(lambda x,y:x*y,vec2classify*p0v)*(1-pA0)
    print('p1:',p1)
    print('p0:',p0)
    if p1>p0:
        return 1
    else:return 0

#朴素贝叶斯测试函数

def testingNB(testVec):
    dataset,classVec = loadDataSet()
    vocablist = creatVocabList(dataset)
    trainMat = get_trainMat(dataset)
    p1v,p0v,pA0 = trainNB(trainMat,classVec)
    thisone = setOfWords2Vec(vocablist,testVec)
    if classifyNB(thisone,p1v,p0v,pA0) == 0:
        print(testVec,'属于非侮辱类') 
    else:
        print(testVec,'属于侮辱类') 
        
testVec1 = ['love','my','dalmation']
testingNB(testVec1)
testVec2 = ['garbage','dog']
testingNB(testVec2)

###################朴素贝叶斯改进之拉普拉斯平滑 
#问题1 ： P(W0|1)P(W1|1)P(W2|1) 其中任何一个为0，乘积也为0
#解决 ：拉普拉斯平滑：将所有词的初始频数设为1，分母设为2
#问题2 ： P(W0|1)P(W1|1)P(W2|1) 每个都太小，数据下溢出
#解决 ： 对乘积结果取对数

#朴素贝叶斯分类器训练函数 改进版

def trainNB2(trainMat,classVec):
    n = len(trainMat)   #总文档数目
    m = len(trainMat[0]) #所有文档中非重复词条数
    pA0 = sum(classVec)/n  #侮辱性文档占总文档的概率
    p0num = np.ones(m)   # 初始化 1
    p1num = np.ones(m)
    p1demo = 2         #分母设为2
    p0demo = 2
    for i in range(n):
        if classVec[i]==1:       
            p1num += trainMat[i] # 侮辱性文档中词条的分布
            p1demo += sum(trainMat[i]) #侮辱性文档中词条总数
        else:
            p0num += trainMat[i]
            p0demo += sum(trainMat[i])
    p1v = np.log(p1num/p1demo)  #侮辱类的条件概率数组取对数
    p0v = np.log(p0num/p0demo) 
    return p1v,p0v,pA0 

p1v,p0v,pA0 = trainNB2(trainMat,classVec)

#测试朴素贝叶斯分类器
 
from functools import reduce

def classifyNB2(vec2classify,p1v,p0v,pA0): # vec2classify 待分类的词条分布数组
    p1 = sum(vec2classify*p1v)+np.log(pA0)  # 原本的连乘取对数变成连加
    p0 = sum(vec2classify*p0v)+np.log(1-pA0)
    print('p1:',p1)
    print('p0:',p0)
    if p1>p0:
        return 1
    else:return 0

#朴素贝叶斯测试函数

def testingNB2(testVec):
    dataset,classVec = loadDataSet()
    vocablist = creatVocabList(dataset)
    trainMat = get_trainMat(dataset)
    p1v,p0v,pA0 = trainNB2(trainMat,classVec)
    thisone = setOfWords2Vec(vocablist,testVec)
    if classifyNB2(thisone,p1v,p0v,pA0) == 0:
        print(testVec,'属于非侮辱类') 
    else:
        print(testVec,'属于侮辱类') 





#####################朴素贝叶斯之垃圾邮件过滤（手动)

import pandas as pd
import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch04\\email')

def get_dataset():
    ham = []
    for i in range(1,26):
        filepath = 'ham/%d.txt'%(i)
        #print(filepath)
        data = open(filepath,encoding = 'gbk',errors = 'ignore').read()
        ham.append([data,'ham'])
    df1 = pd.DataFrame(ham)
    #df11 = df1[:3]
    spam = []
    for i in range(1,26):
        filepath = 'spam/%d.txt'%(i)
        #print(filepath)
        data = open(filepath,encoding = 'gbk',errors = 'ignore').read()
        spam.append([data,'spam'])
    df2 = pd.DataFrame(spam)
    #df22 = df2[:3]
    dataset = pd.concat([df1,df2],ignore_index = True) #合并，忽略df2的索引，默认axis=0,按行合并
    return dataset

dataset = get_dataset() #得到一个数据框

mail = dataset.iloc[:,0]
classVec = dataset.iloc[:,1]
classVec = list(classVec)
  
#切分训练级和测试集

import random

def randsplit(dataset,rate):
    l=list(dataset.index)
    random.shuffle(l)
    dataset.index=l
    n=dataset.shape[0]
    m=int(n*0.8)
    train=dataset.loc[range(m),:] #注意不能写成.iloc or [:m,:]
    test=dataset.loc[range(m,n),:]
    test.index=range(n-m)
    dataset.index=range(n)
    return train,test

train,test=randsplit(mail,0.8)

#获取词条集合dataset

import re 

def get_words(mail):
    dataset = []
    for i in range(mail.shape[0]):
        file = mail[i]
        wordvec = re.split(r'[\W*]',file)
        dataset.append(wordvec)
    #dataset = set(dataset)  错，获取集合和 ‘去重 ’要分开进行
    return dataset
  
dataset = get_words(train) 
        
# 得到无重复词条   

def unique(dataset):
    vocablist = set()
    for doc in dataset:
        vocablist = vocablist|set(doc)
    vocablist = list(vocablist)   # list(set)可，set(list),set(set)不可
    return vocablist

vocablist = unique(dataset)[1:]  #第一个是空格

#获得训练集向量

def get_train(dataset,vocablist):
    n = len(dataset)
    result=[]
    for i in range(n):
        returnvec = [0]*len(vocablist)
        for word in dataset[i]:
            if word in vocablist:
                returnvec[vocablist.index(word)] += 1
        result.append(returnvec)
    return result

trainmat = get_train(dataset,vocablist)

#朴素贝叶斯分类器训练函数

from functools import reduce
import numpy as np

def classifynb(trainmat,classVec):
    n = len(trainmat)
    m = len(trainmat[0])
    PA = classVec.count('spam') #垃圾邮件的概率
    pa1 = np.ones(m)
    pa0 = np.ones(m)
    pb1 = 2
    pb0 = 2
    for i in range(n):
        if classVec =='spam':
            pa1 += trainmat[i]
            pb1 += sum(trainmat[i])
        else:
            pa0 += trainmat[i]
            pb0 += sum(trainmat[i])
    p1v = np.log(pa1/pb1)
    p0v = np.log(pa0/pb0)
    return p1v,p0v,PA
    
 p1v,p0v,PA =  classifynb(trainmat,classVec)  
    
#测试朴素贝叶斯分类器

def classifynb(testvec,p1v,p0v,PA):
    p1 = sum(testvec*p1v) + np.log(PA)  #未分类词组的特征矩阵X条件概率矩阵
    p0 = sum(testvec*p0v) + np.log(1-PA)
    if p1 > p0:
        return 'spam'
    else:
        return 'ham'
    
result = classifynb(testvec,p1v,p0v,PA)  
  
#朴素贝叶斯测试函数  

def test_nb():
    predata = get_dataset() #得到一个数据框
    train,test=randsplit(predata,0.8)
    classVec = dataset.iloc[:,1]
    classVec = list(classVec)              
    train = train.iloc[:,0]
    dataset = get_words(train) 
    vocablist = unique(dataset)[1:]  
    trainmat = get_train(dataset,vocablist)
    p1v,p0v,PA =  classifynb(trainmat,classVec) 
    result=[]
    for i in range(test.shape[0]):
        testvec = get_train(test[i],vocablist)
        returnvec = classifynb(testvec,p1v,p0v,PA) 
        result.append(returnvec)
    retest = test.copy()                    #这样程序执行完毕后，test不改变
    retest['predict'] = result
    accuracy = (retest.iloc[:,-1]==retest.iloc[:,-2]).mean()
    return f'模型的预测准确率为{accuracy}'


#####################朴素贝叶斯之垃圾邮件过滤（sklearn)

import pandas as pd
import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch04\\email')

def get_dataset():
    ham = []
    for i in range(1,26):
        filepath = 'ham/%d.txt'%(i)
        #print(filepath)
        data = open(filepath,encoding = 'gbk',errors = 'ignore').read()
        ham.append([data,'ham'])
    df1 = pd.DataFrame(ham)
    spam = []
    for i in range(1,26):
        filepath = 'spam/%d.txt'%(i)
        #print(filepath)
        data = open(filepath,encoding = 'gbk',errors = 'ignore').read()
        spam.append([data,'spam'])
    df2 = pd.DataFrame(spam)
    dataset = pd.concat([df1,df2],ignore_index = True) #合并，忽略df2的索引，默认axis=0,按行合并
    return dataset

#对文本信息进行特征值抽取  
from sklearn.feature_extraction.text import TfidfVectorizer
###  TfidfVectorizer =  TfidfTransformer + CountVectorizer 
###  TfidfTransformer  把计数矩阵转化成标准化的tf或tf-idf
###  CountVectorizer  把文本文档转化成计数矩阵
#Tf 词频，词语在文档中出现的频率  idf 逆文档频率  
dataset=get_dataset()
tf = TfidfVectorizer()
tf.fit(dataset[0])  # dataset 第0列，对邮件内容进行学习
data_tf = tf.transform(dataset[0])  #对学习的内容进行特征抽取
#data_tf = tf.fit_transform(dataset[0])  #训练和转换可以同时进行

#切分训练集和测试集
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_tf,dataset[1],test_size=0.2)
Xtest.shape[0]
Ytest

#####训练模型

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
#多项式朴素贝叶斯
mnb = MultinomialNB()  #获取模型
mnb.fit(Xtrain,Ytrain) #训练模型
mnb.score(Xtest,Ytest) #查看准确率
#伯努利分布贝叶斯
bnb = BernoulliNB()  #获取模型
bnb.fit(Xtrain,Ytrain) #训练模型
bnb.score(Xtest,Ytest) #查看准确率

#####交叉验证

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['Simhei'] #显示中文
#进行10次十折交叉验证
#将数据集分成十份，轮流将其中9份作为训练数据，1份作为测试数据，精度取10次结果的均值，为1次十折交叉验证
bnbs=[]
for i in range(10):
    mnb = MultinomialNB()
    mnb_s = cross_val_score(mnb,data_tf,dataset[1],cv=10).mean()
    mnbs.append(mnb_s)
    bnb = BernoulliNB()
    bnb_s = cross_val_score(bnb,data_tf,dataset[1],cv=10).mean()
    bnbs.append(bnb_s)

plt.plot(range(1,11),mnbs,label='多项式朴素贝叶斯')
plt.plot(range(1,11),bnbs,label='伯努利朴素贝叶斯')
plt.legend()
plt.show()


######################### Kaggle比赛之旧金山犯罪率预测

import os
os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch04\\Kaggle')

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

train = pd.read_csv('train.csv',parse_dates = ['Dates'])  #日期型数据
test = pd.read_csv('test.csv',parse_dates = ['Dates'],index_col = 0) #把第0列作为索引

#category 犯罪类型（标签），descript 犯罪描述，PdDistrict 所属警区，resolution处理位置，
#address 地址，x and y GPS坐标

#特征预处理
##1 对犯罪类别，用 LabelEncoder 进行编号
leCrime = LabelEncoder()
crime = leCrime.fit_transform(train.Category) #39种犯罪类型
days = pd.get_dummies(train.DayOfWeek) # 因子化星期几的特征（变成0/1列表） 哑变量
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour    #提取小时数，也可 dt.minute/dt.second
hour = pd.get_dummies(hour)

#组合特征形成训练集
trainData = pd.concat([hour,days,district],axis = 1) #crime 和其他的type不一样,不能直接concat
trainData['crime'] = crime

#得到测试集
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = test.Dates.dt.hour    
hour = pd.get_dummies(hour)
testData = pd.concat([hour,days,district],axis = 1) #crime 和其他的type不一样,不能直接concat

#切分数据集
X_train,X_test,Y_train,Y_test = train_test_split(trainData.iloc[:,:-1],trainData.iloc[:,-1],test_size=0.2)

#训练模型
BNB = BernoulliNB()
BNB.fit(X_train,Y_train)

#计算损失函数
propa = BNB.predict_proba(X_test) #proba导出的是每个样本属于某一类的概率
logLoss = log_loss(Y_test,propa)
logLoss

#使用模型预测testData
BNB.predict(testData)













