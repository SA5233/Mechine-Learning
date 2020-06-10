# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:24:26 2020

@author: 韩琳琳
"""

import random

print('*'*30)
print('欢迎进入双色球小程序')
print('*'*30)
numlist=[]
for i in range(1,7):
    n=input('红球%s号码：'%(i))
    if len(n)==2 and int(n)>0 and int(n)<34:
        numlist.append(n)
    else:
        print('号码有误，请重新输入')
        break 
m=input('蓝球号码：')
if len(m)==2 and int(m)>0 and int(m)<17:
    numlist.append(m)
answer = input(f'号码已选定，您的号码为{numlist}，是否确定提交？[y/n]')
if answer=='y':
    r=b=0
    lucky=[]
    for i in range(6):
        num=random.randint(0,33)
        while num in lucky:
            num=random.randint(0,33)           
        lucky.append(num)
    for n in numlist[:-1]:
        for l in lucky:
            if int(n)==l:
                r+=1
    b_lucky=random.randint(0,16)
    lucky.append(b_lucky)
    if int(numlist[-1])==b_lucky:
        b+=1
    print(f'本次中奖号码为{lucky}')
    if r!=0 or b!=0:
        print(f'恭喜您，中{r}+{b}')
    else:
        print('谢谢惠顾，小赌怡情，祝您生活愉快')
if answer=='n':
    print('退出游戏，祝您生活愉快')







