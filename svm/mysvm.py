# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:09:28 2019

@author: 韩琳琳
"""

import numpy as np
import pandas as pd
import os 

os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch06')



**************************** SOV算法 ********************************************************

#构建辅助函数
##生成特征矩阵和标签矩阵
def loadDataSet(file):
    dataSet = pd.read_csv(file,sep='\t',header=None)
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat

file = 'testSet.txt'
xMat,yMat = loadDataSet(file)

##数据集可视化
import matplotlib.pyplot as plt
%matplotlib inline

def showDataSet(xMat,yMat):
    data_p = []     #正样本, 100个1x2矩阵的列表
    data_n = []     #负样本
    m = xMat.shape[0]
    for i in range(m):
        if yMat[i] > 0:
            data_p.append(xMat[i])
        else:
            data_n.append(xMat[i])
    data_pa = np.array(data_p)           #把矩阵列表转成numpy数组
    data_na = np.array(data_n)
    #提取不同类的点的X1,X2特征作为横纵坐标，做散点图
    #data_pa : len=46 , data_pa.T : len=2 ,X1,X2的值各成一数组
    plt.scatter(data_pa.T[0],data_pa.T[1])
    plt.scatter(data_na.T[0],data_na.T[1])

##随机选择alpha对
import random

def selectj(i,m):
    j = i
    while j==i:
        j = int(random.uniform(0,m))
        return j
    
## 修剪 alpha_j
def clipAlpha(aj,H,L):   #使aj处于[L,H]之间
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj



**************************** 简化版 SMO 算法 ********************************************************

# xMat:特征向量，yMat：标签向量，C:惩罚因子，toler:容错率，maxIter:最大迭代次数
def smoSimple(xMat,yMat,C,toler,maxIter):   
    b=0
    m,n=xMat.shape
    alpha = np.mat(np.zeros((m,1)))   #初始化α参数，设为0
    iters = 0                         #初始化迭代次数
    while (iters<maxIter):
        alpha_ = 0                      #初始化α优化次数
        for i in range(m):
            #步骤1：计算误差Ei
            fxi = np.multiply(alpha,yMat).T*(xMat*xMat[i,:].T)+b
            Ei = fxi - yMat[i]
            #优化α，设定容错率
            # 非if abs(yMat[i]*Ei)>toler and (0<alpha[i]<C)
            if ((yMat[i]*Ei<-toler)and(alpha[i]<C)) or ((yMat[i]*Ei>toler)and(alpha[i]>0)):  
                #随机选择一个与alpha_i成对优化的alpha_j
                j = selectj(i,m)
                #步骤1：计算误差Ei
                fxj = np.multiply(alpha,yMat).T*(xMat*xMat[j,:].T)+b
                Ej = fxj - yMat[j]
                #保存更新前的alpha_i,alpha_j
                alphaiold = alpha[i].copy()
                alphajold = alpha[j].copy()
                #步骤2：计算上下界 H,L
                if (yMat[i]!=yMat[j]):
                    L = max(0,alpha[j]-alpha[i])
                    H = min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0,alpha[j]+alpha[i]-C)
                    H = min(C,alpha[j]+alpha[i])
                if L==H:
                    print('L==H')
                    continue
                #步骤3：计算学习率eta(eta是alpha_j的最优修改量)
                eta = 2*xMat[i,:]*xMat[j,:].T - xMat[i,:]*xMat[i,:].T - xMat[j,:]*xMat[j,:].T
                if eta >= 0:
                    print('eta>=0')
                    continue
                #步骤4：更新 alpha_j
                alpha[j] -= yMat[j]*(Ei-Ej)/eta
                #步骤5：修剪 alpha_j
                alpha[j] = clipAlpha(alpha[j],H,L)
                if abs(alpha[j]-alphajold)<0.00001:
                    print('alpha_j 变化太小')
                    continue
                #步骤6：更新 alpha_i
                alpha[i] += yMat[j]*yMat[i]*(alphajold-alpha[j])
                #步骤7： 更新 b_1 和 b_2
                b1 = b-Ei- yMat[i]*(alpha[i]-alphaiold)*xMat[i,:]*xMat[i,:].T - yMat[j]*(alpha[j]-alphajold)*xMat[i,:]*xMat[j,:].T
                b2 = b-Ej- yMat[i]*(alpha[i]-alphaiold)*xMat[i,:]*xMat[j,:].T - yMat[j]*(alpha[j]-alphajold)*xMat[j,:]*xMat[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0<alpha[i])and(C>alpha[i]): b=b1
                elif (0<alpha[j])and(C>alpha[j]): b=b2
                else: b=(b1+b2)/2
                #统计优化次数
                alpha_ += 1
                print(f'第{iters}次迭代,样本{i}，alpha优化次数{alpha_}')
        #更新迭代次数
        if alpha_==0:iters+=1
        else: iters=0
        print(f'迭代次数为:{iters}')
    return b,alpha

%time b,alpha = smoSimple(xMat,yMat,0.6,0.0001,5)

###支持向量的可视化
def get_sv(xMat,yMat,alpha):
    m = xMat.shape[0]
    sv_x = []
    sv_y = []
    for i in range(m):
        if alpha[i]>0:
            sv_x.append(xMat[i])
            sv_y.append(yMat[i])
    sv_x1 = np.array(sv_x).T
    sv_y1 = np.array(sv_y).T
    return sv_x1,sv_y1

def showplot(xMat,yMat,alpha,b):
    data_p = []     #正样本
    data_n = []     #负样本
    m = xMat.shape[0]
    for i in range(m):
        if yMat[i] > 0:
            data_p.append(xMat[i])
        else:
            data_n.append(xMat[i])
    data_pa = np.array(data_p)           #把矩阵列表转成numpy数组
    data_na = np.array(data_n)
    #提取不同类的点的X1,X2特征作为横纵坐标，做散点图
    #data_pa : len=46 , data_pa.T : len=2 ,X1,X2的值各成一数组
    plt.scatter(data_pa.T[0],data_pa.T[1])
    plt.scatter(data_na.T[0],data_na.T[1])
    #绘制支持向量
    sv_x,sv_y = get_sv(xMat,yMat,alpha)
    plt.scatter(sv_x[0],sv_x[1],s=150,c='none',alpha=0.7,linewidth=1.5,edgecolor='green')
    #绘制超平面
    #reshape(-1,1) 不知道几行，确定是1列
    # np.tile(b, (2, 1)) 沿X轴复制1倍（相当于没有复制），再沿Y轴复制2倍
    #              arrey             matrix
    # multiply  对应元素相乘        对应元素相乘 
    #    *      对应元素相乘         矩阵乘法
    #   dot      矩阵乘法            矩阵乘法
    w = np.dot((np.tile(np.array(yMat).reshape(-1,1),(1,2))*np.array(xMat)).T,np.array(alpha))  
    a1,a2 = w
    x1 = max(xMat[:,0])[0,0]
    x2 = min(xMat[:,0])[0,0]
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1,y2 = (-b-a1*x1)/a2 , (-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    plt.show()
  
    

**************************** 完整版 SMO 算法 ********************************************************

#数据结构
class optStruct:
    def __init__(self,xMat,yMat,C,toler):
        self.X = xMat
        self.Y = yMat
        self.C = C          #松弛变量
        self.tol = toler    #容错率
        self.m = xMat.shape[0]
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eca = np.mat(np.zeros((self.m,2)))  #初始化误差缓存，第一列为是否有效的标志位，第二列是实际误差E的值
        
#计算误差
# os 数据结构 ， k 标号为k的数据 ，Ek 标号为k的数据误差
def calcEk(os,k):
    fxk = np.multiply(os.alpha,os.Y).T*(os.X*os.X[k,:].T) + os.b
    Ek = fxk - os.Y[k]
    return Ek

#内循环启发方式
# i 标号为i的数据的索引值， j,maxk 标号为 j或maxk的数据的索引值
def selectJ(i,os,Ei):
    maxk = -1
    maxDeltaE = 0
    Ej = 0
    os.eca[i] = [1,Ei]              #根据Ei更新误差缓存
    eca = np.nonzero(os.eca[:,0].A)[0] #返回误差不为0的数据的索引值
    if (len(eca))>1:                    #有不为0的误差
        for k in eca:
            if k==i:continue           #不计算i 浪费时间 deltaE=0
            Ek = calcEk(os,k)
            deltaE = abs(Ei-Ek)
            if (deltaE>maxDeltaE):
                maxk = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxk,Ej
    else:
        j = selectj(i,os.m)
        Ej = calcEk(os,j)
    return j,Ej

#计算Ek,并更新误差缓存
def updateEk(os,k):
    Ek = calcEk(os,k) 
    os.eca[k] = [1,Ek]       #更新误差缓存
            
#### 寻找决策边界的优化例程
# 返回1 有任意一对alpha值发生变化， 0 没有
def innerL(i,os):
    #步骤1：计算误差Ei
    Ei = calcEk(os,i)
    #优化α，设定容错率
    if((os.Y[i]*Ei<-os.tol) and (os.alpha[i]<os.C)) or ((os.Y[i]*Ei>os.tol) and (os.alpha[i]>0)):
        #使用内循环启发方式选择alpha_j,并计算Ej
        j,Ej = selectJ(i,os,Ei)
        #保存更新前的alpha值，使用深拷贝
        alphaiold = os.alpha[i].copy()
        alphajold = os.alpha[j].copy()
        #步骤2：计算上下界 H,L
        if (os.Y[i]!=os.Y[j]):
            L = max(0,os.alpha[j]-os.alpha[i])
            H = min(os.C,os.C+os.alpha[j]-os.alpha[i])
        else:
            L = max(0,os.alpha[j]+os.alpha[i]-os.C)
            H = min(os.C,os.alpha[j]+os.alpha[i])
        if L==H:
            #print('L==H')
            return 0
        #步骤3：计算学习率eta(eta是alpha_j的最优修改量)
        eta = 2*os.X[i,:]*os.X[j,:].T - os.X[i,:]*os.X[i,:].T - os.X[j,:]*os.X[j,:].T
        if eta >= 0:
            #print('eta>=0')
            return 0
        #步骤4：更新 alpha_j
        os.alpha[j] -= os.Y[j]*(Ei-Ej)/eta
        #步骤5：修剪 alpha_j
        os.alpha[j] = clipAlpha(os.alpha[j],H,L)
        updateEk(os,j)
        if abs(os.alpha[j]-alphajold)<0.00001:
            #print('alpha_j 变化太小')
            return 0
        #步骤6：更新 alpha_i
        os.alpha[i] += os.Y[j]*os.Y[i]*(alphajold-os.alpha[j])
        updateEk(os,i)
        #步骤7： 更新 b_1 和 b_2
        b1 = os.b-Ei- os.Y[i]*(os.alpha[i]-alphaiold)*os.X[i,:]*os.X[i,:].T - os.Y[j]*(os.alpha[j]-alphajold)*os.X[i,:]*os.X[j,:].T
        b2 = os.b-Ej- os.Y[i]*(os.alpha[i]-alphaiold)*os.X[i,:]*os.X[j,:].T - os.Y[j]*(os.alpha[j]-alphajold)*os.X[j,:]*os.X[j,:].T
        #步骤8：根据b_1和b_2更新b
        if (0<os.alpha[i])and(os.C>os.alpha[i]): os.b=b1
        elif (0<os.alpha[j])and(os.C>os.alpha[j]): os.b=b2
        else: os.b=(b1+b2)/2
        return 1
    else:
        return 0



**************************** 完整线性 SMO 算法 ********************************************************

def smop(xMat,yMat,C,toler,maxIter):
    os = optStruct(xMat,yMat,C,toler)
    iters = 0           #初始化当前迭代次数
    entireSet = True
    alpha_ = 0         #初始化α优化次数
    #迭代次数小于最大迭代次数 且 优化次数>0 或 迭代次数小于最大迭代次数 且 entireSet=True 进入循环
    while(iters<maxIter) and ((alpha_>0) or (entireSet)):
        alpha_ = 0
        if entireSet:            #遍历整个数据集
            for i in range(os.m):
                alpha_ += innerL(i,os)
            iters += 1
        else:            # 此时 entireSet = False 遍历不在边界 0 和 C 的 alpha
            nonBoundis = np.nonzero((os.alpha.A>0)*(os.alpha.A<C))[0]
            for i in nonBoundis:
                alpha_ += innerL(i,os)
            iters += 1
        if entireSet:
            entireSet = False
        elif (alpha_ == 0):         # if entireSet = False and alpha_ == 0
            entireSet = True
    return os.b,os.alpha

%time b,alpha = smop(xMat,yMat,0.6,0.0001,40)

######计算模型准确率
# 计算 w
def calcws(alpha,xMat,yMat):
    m,n = xMat.shape
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alpha[i]*yMat[i],xMat[i,:].T) # alpha[i]*yMat[i] 两个1X1矩阵相乘
        # W += (alpha[i]*yMat[i]*xMat[i,:]).T
    return w                       # mX1 矩阵

#计算准确率
def calcAcc(xMat,yMat,w,b):
    yhat=[]
    re=0
    m,n = xMat.shape
    for i in range(m):
        result = xMat[i]*np.mat(w) + b  #超平面计算公式
        if result<0:
            yhat.append(-1)
        else:
            yhat.append(1)
        if yhat[i]==yMat[i]:
            re +=1
    acc = re/m
    print(f'模型预测准确率为{acc}')
    return acc

w =  calcws(alpha,xMat,yMat)
calcAcc(xMat,yMat,w,b)
showplot(xMat,yMat,alpha,b)


**************************** 非线性SVM算法 ********************************************************

xMat,yMat = loadDataSet('testSetRBF.txt')
showDataSet(xMat,yMat)

xMat,yMat = loadDataSet('testSetRBF2.txt')
showDataSet(xMat,yMat)

####构建核转换函数
def kernelTrans(X,A,ktup):   #ktup 包含核函数信息的元组
    m,n = X.shape
    K = np.mat(np.zeros((m,1)))
    if ktup[0] == 'lin':
        K = X * A.T            #线性核函数，只进行内积
    elif ktup[0] == 'rbf':    #高斯核函数
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T 
        K = np.exp(K/(-1*ktup[1]**2))
    else:
        raise NameError('核函数无法识别')
    return K

#数据结构
class optStruct:
    def __init__(self,xMat,yMat,C,toler,ktup):
        self.X = xMat
        self.Y = yMat
        self.C = C          #松弛变量
        self.tol = toler    #容错率
        self.m = xMat.shape[0]
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eca = np.mat(np.zeros((self.m,2)))  #初始化误差缓存，第一列为是否有效的标志位，第二列是实际误差E的值
        self.K = np.mat(np.zeros((self.m,self.m))) #初始化核K, np.zeros 数组
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],ktup)
            

#计算误差
# os 数据结构 ， k 标号为k的数据 ，Ek 标号为k的数据误差
def calcEk2(os,k):
    fxk = np.multiply(os.alpha,os.Y).T*os.K[:,k] + os.b
    Ek = fxk - os.Y[k]
    return Ek


#### 寻找决策边界的优化例程 2 
# 返回1 有任意一对alpha值发生变化， 0 没有
def innerL2(i,os):
    #步骤1：计算误差Ei
    Ei = calcEk2(os,i)
    #优化α，设定容错率
    if((os.Y[i]*Ei<-os.tol) and (os.alpha[i]<os.C)) or ((os.Y[i]*Ei>os.tol) and (os.alpha[i]>0)):
        #使用内循环启发方式选择alpha_j,并计算Ej
        j,Ej = selectJ(i,os,Ei)
        #保存更新前的alpha值，使用深拷贝
        alphaiold = os.alpha[i].copy()
        alphajold = os.alpha[j].copy()
        #步骤2：计算上下界 H,L
        if (os.Y[i]!=os.Y[j]):
            L = max(0 , os.alpha[j]-os.alpha[i])
            H = min(os.C , os.C+os.alpha[j]-os.alpha[i])
        else:
            L = max(0 , os.alpha[j]+os.alpha[i]-os.C)
            H = min(os.C , os.alpha[j]+os.alpha[i])
        if L==H:
            #print('L==H')
            return 0
        #步骤3：计算学习率eta(eta是alpha_j的最优修改量)
        eta = 2*os.K[i,j] - os.K[i,i] - os.K[j,j]
        if eta >= 0:
            #print('eta>=0')
            return 0
        #步骤4：更新 alpha_j
        os.alpha[j] -= os.Y[j]*(Ei-Ej)/eta
        #步骤5：修剪 alpha_j
        os.alpha[j] = clipAlpha(os.alpha[j],H,L)
        updateEk(os,j)
        if abs(os.alpha[j]-alphajold)<0.00001:
            #print('alpha_j 变化太小')
            return 0
        #步骤6：更新 alpha_i
        os.alpha[i] += os.Y[j]*os.Y[i]*(alphajold-os.alpha[j])
        updateEk(os,i)
        #步骤7： 更新 b_1 和 b_2
        b1 = os.b-Ei- os.Y[i]*(os.alpha[i]-alphaiold)*os.K[i,i] - os.Y[j]*(os.alpha[j]-alphajold)*os.K[i,j]
        b2 = os.b-Ej- os.Y[i]*(os.alpha[i]-alphaiold)*os.K[i,j] - os.Y[j]*(os.alpha[j]-alphajold)*os.K[j,j]
        #步骤8：根据b_1和b_2更新b
        if (0<os.alpha[i])and(os.C>os.alpha[i]): os.b=b1
        elif (0<os.alpha[j])and(os.C>os.alpha[j]): os.b=b2
        else: os.b=(b1+b2)/2
        return 1
    else:
        return 0 
    
    
    
**************************** 完整线性 SMO 算法2 （加入线性核函数） *********************************************


def smop2(xMat,yMat,C,toler,maxIter,ktup=('lin',0)):
    os = optStruct(xMat,yMat,C,toler,ktup)
    iters = 0           #初始化当前迭代次数
    entireSet = True
    alpha_ = 0         #初始化α优化次数
    #遍历整个数据集
    while(iters<maxIter)and((alpha_>0) or (entireSet)):
        alpha_ = 0
        if entireSet:
            for i in range(os.m):
                alpha_ += innerL2(i,os)
            iters += 1
        else:
            nonBoundis = np.nonzero((os.alpha.A>0)*(os.alpha.A<C))[0]
            for i in nonBoundis:
                alpha_ += innerL2(i,os)
            iters += 1
        if entireSet:
            entireSet = False
        elif (alpha_ == 0):
            entireSet = True
    return os.b,os.alpha
    
%time b,alpha = smop2(xMat,yMat,0.6,0.0001,40,ktup=('lin',0))    
    

#############利用核函数进行分类 

def testRbf(k1 = 1.3):
    xMat,yMat = loadDataSet('testSetRBF.txt')
    b,alpha = smop2(xMat,yMat,200,0.0001,100,('rbf',k1))
    svind = np.nonzero(alpha.A>0)[0]  #返回两个数组分别给出alpha非零元素行列的索引值
    svs = xMat[svind]
    labelsv = yMat[svind]
    print(f'支持向量个数：{svs.shape[0]}')
    m,n = xMat.shape
    errorcount = 0
    for i in range(m):
        K = kernelTrans(svs,xMat[i,:],('rbf',k1))  #计算各个点的核
        predict = K.T * np.multiply(labelsv,alpha[svind]) + b #根据支持向量的点，预测结果
        if np.sign(predict) != np.sign(yMat[i]):  #sign 返回 -1，0，1
            errorcount += 1
    acc_train = 1-errorcount/m
    print(f'训练集准确率为：{acc_train}')
    xMat2,yMat2 = loadDataSet('testSetRBF2.txt')
    m2,n2 = xMat2.shape
    errorcount2 = 0
    for i in range(m2):
        K = kernelTrans(svs,xMat2[i,:],('rbf',k1))  #此处svs 是根据训练集得出的
        predict2 = K.T * np.multiply(labelsv,alpha[svind]) + b  #根根据训练集得出的
        if np.sign(predict2) != np.sign(yMat2[i]):  #sign 返回 -1，0，1
            errorcount2 += 1
    acc_test = 1-errorcount2/m
    print(f'测试集准确率为：{acc_test}')
    return acc_train,acc_test
 
    
##保持k1=1.3的情况下，绘制运行10次的结果
import matplotlib.pyplot as plt   
%matplotlib inline
plt.rcParams['font.sans-serif']=['simhei']

train_acc=[]
test_acc=[]
for i in range(10):
    a,b = testRbf(k1=1.3)
    train_acc.append(a)
    test_acc.append(b)
plt.plot(range(1,11),train_acc)  #横纵坐标
plt.plot(range(1,11),test_acc)   
plt.xlabel('次数') 
plt.ylabel('模型准确率') 
plt.legend(['train_acc','test_acc'])
plt.show()
    
#####测试K取值的效果
train_acc=[]  
test_acc=[] 
for k1 in np.arange(0.1,1.4,0.1) : #返回从0.1到1.4，间隔0.1的数组
    a,b = testRbf(k1)
    train_acc.append(a)
    test_acc.append(b)
plt.plot(np.arange(0.1,1.4,0.1),train_acc)  #横纵坐标
plt.plot(np.arange(0.1,1.4,0.1),test_acc)   
plt.xlabel('K1') 
plt.ylabel('模型准确率') 
plt.legend(['train_acc','test_acc'])
plt.show()
    
    
***************************** SVM 之手写数字识别 ***************************************************************    
    
def get_Mat(path) :
    filelist = os.listdir(path)  #提取文件夹中所有文件的名字
    m = len(filelist)
    label = []
    xMat = np.mat(np.zeros((m,1024)))
    for i in range(m):
        #xMat_i = np.mat(np.zeros((1,1024)))
        filename = filelist[i]
        #读取文件，结果是一个32*32的dataframe
        txt = pd.read_csv(f'{path}/{filename}',header=None)
        numlist = []
        for j in range(32):
            num=txt.iloc[j,:]
            numlist.extend(num[0])  #单个添加，得到1024个数字
        xMat[i] = numlist
# =============================================================================
#      官方答案：
#       for j in range(32):
#             num = txt.iloc[j,:]
#             for k in range(32):
#                 xMat_i[0,32*j+k] = int(num[0][k])
#         xMat[i,:] = xMat_i
# =============================================================================
# =============================================================================
#       这样也不行，一个1024数列赋给1X1024的矩阵会溢出
#        x=''
#         for j in range(32):
#             num=txt.iloc[j,:]
#             numlist.append(num[0])
#         x = x.join(numlist)
#         xMat[i] = np.array(x)        
# =============================================================================
        filelabel = int(filename.split('_')[0])
        if filelabel == 9:
            label.append(-1)
        else:
            label.append(1)
    yMat = np.mat(label).T
    return xMat,yMat

os.chdir('C:\\Users\\SA\\Documents\\machine learning\\code\\Ch06')
path='digits\\trainingDigits'
xMat,yMat = get_Mat(path)    
    
##### 手写数字的测试函数
def testDigits(ktup=('rbf',10)):
    xMat,yMat = get_Mat('digits\\trainingDigits') 
    b,alpha = smop2(xMat,yMat,200,0.0001,100 ,ktup)
    svind = np.nonzero(alpha.A>0)[0]
    svs = xMat[svind]
    labelsv = yMat[svind]
    print(f'支持向量个数：{svs.shape[0]}')
    m,n = xMat.shape
    errorcount = 0
    for i in range(m):
        K = kernelTrans(svs,xMat[i,:],ktup) #进行数据转换
        predict = K.T * np.multiply(labelsv,alpha[svind]) + b
        if np.sign(predict) != np.sign(yMat[i]):
            errorcount += 1
    acc_train = 1- errorcount/m
    print(f'训练集准确率为：{acc_train}')
    xMat2,yMat2 = get_Mat('digits\\testDigits')
    m2,n2 = xMat2.shape
    errorcount2 = 0
    for i in range(m2):
        K = kernelTrans(svs,xMat2[i,:],ktup)  #此处svs 是根据训练集得出的
        predict2 = K.T * np.multiply(labelsv,alpha[svind]) + b  #根根据训练集得出的
        if np.sign(predict2) != np.sign(yMat2[i]):  #sign 返回 -1，0，1
            errorcount2 += 1
    acc_test = 1-errorcount2/m
    print(f'测试集准确率为：{acc_test}')
    return acc_train,acc_test,svs.shape[0]
    
    
#不同的核函数及参数运行效果
acc_train = []  
acc_test = [] 
svnum = [] 
ktups = [('rbf',0.1),('rbf',5),('rbf',10),('rbf',50),('rbf',100),('lin',0)]
for ktup in ktups:
    a,b,c = testDigits(ktup)
    acc_train.append(a)
    acc_test.append(b)
    svm.append(c)
df = pd.DataFrame({'内核设置':ktups,'训练准确率':acc_train,'测试准确率':acc_test,'支持向量数':svnum})
#美观处理
for i in df.columns[1:-1]:
    df.loc[:,i] = [round(x*100,1) for x in df.loc[:,i].values]

df





