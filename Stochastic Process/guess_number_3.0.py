# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:08:26 2020

@author: 韩琳琳
"""
"""幸运数字猜猜猜 3.0-韩琳琳  2020年6月6日"""

import random

print('*'*30)
print('欢迎进入幸运数字猜猜猜')
print('*'*30)

username=input('请输入您的姓名:')
money=0
answer=input('您是否要进入游戏(y/n)?')

while  answer=='y':
        #产生幸运数字
    if money<2:
        try:
            n=int(input('您的金币不足,请输入充值金额(1元10币),也可按任意键退出'))
        except:
            break
        money+=10*n
    money-=2
    print('玩一局游戏扣除2个币,当前游戏币是:{}'.format(money))
    print('进入游戏.............')
    num=random.randint(0,9)
    i=1
    while i<4:
        guess=int(input('请您猜一猜今天的幸运数字是0-9中的哪个数:'))
        if guess==num:
                   print('恭喜您,{}!本局获奖励3个游戏币!'.format(username))
                   money+=3
                   break
        else:
                     if i!=3:
                         if guess>num:
                             print("抱歉您猜大了，再猜一次")
                         else:
                            print("抱歉您猜小了，再猜一次")
                     if i==3:
                        print("抱歉机会已用尽")                       
        i=i+1
    answer= input('是否再来一局，需扣除2个游戏币？(y/n)')
    
if answer !='y':
            print('退出游戏')
            
print('GAME OVER')