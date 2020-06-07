# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:29:24 2019

@author: 韩琳琳
"""

import numpy as np
import pandas as pd

#######海洋生物的香农熵

def ent(dataset):
    n=dataset.shape[0]
    iset=dataset.iloc[:,-1].value_counts()
    p=iset/n
    ent=-(p*np.log2(p)).sum()#无参表示全部相加，axis=1按行相加，axis=0按列相加
    return ent


def createdata_():
    sea={'underwater survive':[1,1,1,0,0],'flippers':[1,0,0,1,1],
         'fish':['yes','yes','no','no','no']}      
    data=pd.DataFrame(sea)
    return data

dataset=createdata_() #熵越大，表明信息的不纯度越高，信息混合度高

###########海洋生物的信息增益

#计算第0列的信息增益
def gain_0(data):
    n=data.shape[0]
    m=data[data['underwater survive']==1]
    n1=m.shape[0]
    p=n1/n
    mm=data[data['underwater survive']==0]
    n2=mm.shape[0]
    pa=n2/n
    return ent(data)-p*ent(m)-pa*ent(mm)   #参数和函数shang()的重复也不影响

gain_0(createdata_())  #0.42 

#计算第1列的信息增益
def gain_1(data):
    n=data.shape[0]
    m=data[data['flippers']==1]
    n1=m.shape[0]
    p=n1/n
    mm=data[data['flippers']==0]
    n2=mm.shape[0]
    pa=n2/n
    return ent(data)-p*ent(m)-pa*ent(mm)   #参数和函数shang()的重复也不影响

gain_1(createdata_())  #0.17
    
###########数据集最佳切分函数
#原则：信息增益最大，代表熵下降最快，即达到叶节点速度最快

def bestsplit(data):
    bent=ent(data)
    basegain=0
    axis=-1
    for i in range(data.shape[1]-1):   #循环列
        levels=data.iloc[:,i].value_counts().index #这一列有几个子节点
        cent=0 #使用变量前要定义，位置要在其外一层
        for j in levels:           #循环列下的节点
            childa=data[data.iloc[:,i]==j] #某一个子节点的dataframe
            p=childa.shape[0]/data.shape[0] #da是series
            cent+= p*ent(childa)
        gain=bent-cent
        if gain>basegain:
            basegain=gain
            axis=i
    return axis
    
bestsplit(data)        

def mysplit(dataset,axis,value):
    col=dataset.columns[axis]
    redata=dataset.loc[dataset[col]==value,:].drop(col,axis=1)
    return redata

mysplit(data,0,0)

#########决策树的构建

def creatree(dataset):
    colist=list(dataset.columns)
    classlist=dataset.iloc[:,-1].value_counts()
    if classlist[0]==dataset.shape[0]:
        return classlist.index[0]
    elif dataset.shape[1]==1:
        return 'unknown'
    axis=bestsplit(dataset)
    valuelist=set(dataset.iloc[:,axis])
    char=colist[axis]
    mytree={char:{}}
    del colist[axis]
    for v in valuelist:
        mytree[char][v]=creatree(mysplit(dataset,axis,v))
    return mytree
mytree=creatree(dataset)
mytree

##树的存储
np.save('mytree.npy',mytree)

  
##树的读取
read_mytree=np.load('mytree.npy',allow_pickle=True)
read_mytree

#######使用决策树执行分类
test=data.iloc[2,:-1]
test
def classify(tree,labels,test):
    firstr=next(iter(tree)) #取决策树的第一个节点
    # firstr=list(tree)[0]
    secdict=tree[firstr]
    col=labels.index(firstr)
    for key in secdict.keys():
        if test[col]==key:
            if type(secdict[key])==dict:
                classlabel=classify(secdict[key],labels,test)
            else: classlabel=secdict[key]
    return classlabel
             
classify(mytree,list(data.columns),test)

#########预测
def acc_classify(train,test):
    intree=creatree(train)
    labels=list(train.columns)
    result=[]
    for i in range(test.shape[0]):
        testvec=test.iloc[i,:-1]
        fish=classify(intree,labels,testvec)
        result.append(fish)
    retest = test.copy()                    #这样程序执行完毕后，test不改变
    retest['predict'] = result
    acc = (retest.iloc[:,-1]==retest.iloc[:,-2]).mean()
    print(f'预测准确度为{acc}')

acc_classify(dataset,dataset)

###############决策树的可视化 

###包 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

dataset=data
#特征
Xtrain=dataset.iloc[:,:-1]
#标签
Ytrain=dataset.iloc[:,-1]
#labels=set(dataset.iloc[:,-1]) #type=set
labels=Ytrain.unique().tolist()
Ytrain=Ytrain.apply(lambda x:labels.index(x)) #将本文转换成数字
#绘制树模型
clf=DecisionTreeClassifier()
clf=clf.fit(Xtrain,Ytrain)
tree.export_graphviz(clf)
dot_data=tree.export_graphviz(clf,out_file=None)
graphviz.Source(dot_data)
#给图形增加标签和颜色
dot_data=tree.export_graphviz(clf,out_file=None,
           feature_names=['underwater surviving','flippers'],
           class_names=['fish','not fish','unknow'],filled=True,rounded=True,
           special_characters=True)
graphviz.Source(dot_data)
graph=graphviz.Source(dot_data)
graph.render('fish')

#####手动可视化

##计算叶子节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree)) 
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key])=='dict':         #是字典就递归
            numLeafs += getNumLeafs(secondDict[key])  #是叶子节点
        else:   numLeafs +=1
    return numLeafs


##树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key])==dict:#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth
   

##绘制节点
    
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] #中文

#nodeTxt节点名 ,centerPt子节点坐标 ，parentPt 父节点坐标，nodeType节点格式 
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    createPlot.ax1.annotate(nodeTxt, 
                            xy=parentPt,  xycoords='axes fraction', #从左下角定义坐标轴
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center",  #箭头从中间到中间
                            bbox=nodeType, #节点类型，方or圆
                            arrowprops=arrow_args )
    

#标注有向边属性值
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  #计算标注位置的横坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]  #计算标注位置的纵坐标
    createPlot.ax1.text(xMid, yMid, txtString, 
                        va="center", ha="center",   #横竖都放中间
                        rotation=30)  #角度


#绘制树
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))    #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#确定中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key])==dict:#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW  #x初始位置偏移
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    
#绘制画布

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #去掉x,y轴
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW  #x的初始偏移
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
    
createPlot(mytree)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
























