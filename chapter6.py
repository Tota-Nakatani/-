#6章 学習に関するテクニック
#パラメータの更新
#確率的勾配降下法(SGD)のデメリットとは・・・
#W=W-η*∂L/∂W
#classとして実装
#このクラスに勾配とパラメータを渡してあげると、最適化を行う！

#____________________________________________________________________________________________________

class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]

#____________________________________________________________________________________________________

#SGDの欠点
#f(x,y)=1/20x^2+y^2の最小値を求める、問題について考える

#関数のプロット
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def function(x,y):
    return (x**2/20)+y**2

x=np.arange(-10,10,0.1)
y=np.arange(-10,10,0.1)
X,Y=np.meshgrid(x,y)
Z=function(X,Y)

fig=plt.figure()
ax=Axes3D(fig)
ax.set_xlabel=("x")
ax.set_ylabel=("y")
ax.set_zlabel=("f(x,y)")
ax.plot_wireframe(X,Y,Z,color='yellow')
plt.show()

#お椀をx軸方向に伸ばしたような形
#y軸方向の勾配は大きいが、x軸方向の勾配は小さい
#最小値は(0,0)であるが、勾配は(0,0)を向かない　p.169の等高線参考
#そのためストレートに最小値に向かわずにジグザグの動きをする
#SGDに変わる最適化法として・・・
#Momentum・AdaGrad・Adam

#____________________________________________________________________________________________________

#Momentum
#v←αv-η*∂L/∂W
#W←W+v
#あらたな変数vを導入する
#ボールがU型の傾斜を転がるイメージ
#αは減衰定数のイメージ

class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=Momentum
        self.v=None

    def update(self,params,grads):
        #ディクショナリ型データとしてvを生成
        if self.v is None:
            self.v={}
            #itemsはkeyとvalueをまとめたもの
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)
        
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
            params[key]+=self.v[key]

#__________________________________________________________________________________________________

#AdaGrad
#学習係数ηの値が重要
#ηを減衰させるテクニックがある
#最初は大きく、次第に小さく
#パラメータの要素ごとに適応的にパラメータを更新する
#h←h+(∂L/∂W)*(∂L/∂W)
#W=W-η*(1/√h)*(∂L/∂W)
#hにより学習率を調節する

class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None

    def update(self,params,grads):
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)

        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)

#____________________________________________________________________________________________________

#Adam
#上記2つのアイデアを組み合わせたような手法

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

#____________________________________________________________________________________________________

#重みの初期値
#小さい値で初期化したいが、0は使わない！
#全て同じ値で伝搬してしまう

#隠れ層のアクティベーション分布
#活性化関数を通した後の出力データを観察する
#sigmoid活性化を行う5層のニューラルネットワークを仮定する

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Relu(x):
    return np.maximum(o,x)

def tanh(x):
    return np.tanh(x)

#100の特徴量を持つ、1000のデータをinputとする
input_data=np.random.randn(1000,100)   
#隠れ層のニューロンの数
node_num=100
hidden_layer_size=5#隠れ層は5層
#ここに活性化された後の値を収納する
activations={}

x=input_data

for i in range(hidden_layer_size):
    if i!=0:
        #x（入力は前層のactivationされた値)
        x=activations[i-1]

    #初期値を変更し、実験する
    #1例として以下の乱数を用いる
    #これは活性関数が線形である時に有効な重みの初期値として有名
    #"Xavierの初期値"
    #前層のノードの数をnとして、
    #乱数分布を√nで正規化する
    w=np.random.randn(node_num,node_num)/np.sqrt(node_num)

    a=np.dot(x,w)

    #活性化関数も色々試そう
    z=sigmoid(a)

    #zをディクショナリに収納する
    activations[i]=z

#ヒストグラムを描写
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i!=0:plt.yticks([],[])
    plt.hist(a.flatten(),30,range=(0,1))
    plt.show()



#Reluの場合、非線形関数であるためまた別の重みの初期値を設定する。
#Heの初期値
#√2/nで初期化