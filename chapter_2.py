#パーセプトロン
#ANDゲートの実装
def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=w1*x1+w2*x2
    if tmp>=theta:
        return 1
    else:
        return 0

AND(0,0)

#重みとバイアスの代入
#y=0(w1x1+w2x2+b<0)
#  1(w1x1+w2x2+b>0)
#b:バイアス、w:重み
import numpy as np 
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    if np.sum(w*x)+b>=0:
        return 1
    else:
        return 0

AND(0,1) 


#

