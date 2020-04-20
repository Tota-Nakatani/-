#ニューラルネットワークの学習
#データから特徴量（ベクトル)を抽出する
#訓練データとテストデータを分ける必要がある・・・汎化性能を持たせるため
#ある１つのデータ群のみに特化してしまう・・・過学習

#損失関数
#MSE:二乗和誤差
#E=1/2(Σ(y-t)^2))
def mean_squard_error(y,t):
    return 0.5*np.sum((y-t)**2)

#正解は2
t=[0,0,1,0,0,0,0,0,0,0]
#2の出力が最も大きい時
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
mean_squard_error(np.array(y),np.array(t))
#7の出力が最も大きい時
y_1=[0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_squard_error(np.array(y_1),np.array(t))

#誤差は小さい方が良い

#交差エントロピー誤差
#E=-Σtlogy
#これも小さい方が良い
#logの中身が０になると発散してしまうのでdeltaを足す
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

#ミニバッチ学習
#600000のデータのなかから無作為に100のバッチを取り出し、それについて学習する
import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)


#np.random.choice(a,b):aからbをランダムに取り出す
train_size=x_train.shape[0]#60000
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
#x_batchは(10,784)
x_batch=x_train[batch_mask]
#t_batchは(10,10)
t_batch=t_train[batch_mask]


#バッチ対応交差エントロピー
#両方のケースに対応できるように
#one-hotラベルの時
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    
    batch_size=y.shape[0]
    return -np.sun(t*np.log(y+1e-7))/batch_size

#ラベルが値として与えられた時
def cross_entropy_error(y,t):
    if y.dim==1:
        t=t.reshape(1,t.size)
        y=t.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size,t)]+1e-7))/batch_size

#微分
#微小な差分により微分を行う：数値微分
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h)/2*h)

#y=0.01x^2+0.1xを実装し、微分する
def function_1(x):
    return 0.01*x**2+0.1*x

import numpy as np
import matplotlib.pylab as plt

numerical_diff(function_1,10)

#偏微分
#y=a^2+b^2

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

def function_2(x,y):
    return x**2+y**2

x=np.arange(-3,3,0.01)
y=np.arange(-3,3,0.01)
X,Y=np.meshgrid(x,y)
Z=function_2(X,Y)
fig=plt.figure()
ax=Axes3D(fig)
ax.set_xlabel=("x")
ax.set_ylabel=("y")
ax.set_zlabel=("f(x,y)")
ax.plot_wireframe(X,Y,Z,color='yellow')
plt.show()


___________________________________________________________________________________


#偏微分と勾配
#インポート
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


#勾配計算(ベクトル)
def numerical_gradient_nobatch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h#str型の数列をfloatに変換
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


#バッチ処理に対応した勾配関数
def numerical_gradient(f,X):
    #一次元(１つのデータの時のみは上記の関数で処理)
    if X.ndim==1:
        return numerical_gradient_nobatch(f,X)
    #複数のデータにも対応する
    else:
        grad=np.zeros_like(X)

        for idx,x in enumerate(X):#enumerate:インデックスも同時に取得
            grad[idx]=numerical_gradient_nobatch(f,x)

        return grad

#関数を定義
def function_2(x):
    if x.ndim==1:
        return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)

#接線を返す関数
def tangent_line(f,x):
    d=numerical_gradient(f,x)
    print(d)
    y=f(x)-d*x
    return lambda t:d*t+y#lambda:引数で関数を返す


if __name__ =='__main__':#気にしない
    x0=np.arange(-2,2,0.25)
    x1=np.arange(-2,2,0.25)
    X,Y=np.meshgrid(x0,x1)

    X=X.flatten()#flatten:多次元配列を一次元に
    Y=Y.flatten()

    grad=numerical_gradient(function_2,np.array([X,Y]).T).T

    plt.figure()
    plt.quiver(X,Y,-grad[0],-grad[1],angles="xy",color="#666666")
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()


_______________________________________________________________________________


#勾配法
#勾配が最も小さくなる方向に少しずつ移動する：勾配降下法
#x=x-η*(∂f/∂x)
#η：学習率
#勾配降下法の実装

#勾配計算関数
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val

    return grad

#勾配降下関数数
def gradient_descent(f,init_x,lr=0.01,epoch=100):
    x=init_x

    for i in range(epoch):
        grad=numerical_gradient(f,x)
        x -=lr*grad

    return x


#関数を定義
def function_2(x):
    return x[0]**2+x[1]**2

init_x=np.array([-3.0,-4.0])
gradient_descent(f=function_2,init_x=init_x,lr=0.1,epoch=100)


________________________________________________________________________________


#ニューラルネットワークに対する勾配
#損失関数の重みに対する勾配を求める
import sys,os
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch04')  # カレントディレクトリをch04に変更
sys.path.append(os.pardir)
import numpy as np
#deeplearnigライブラリーにある.pyファイルから関数をインポート
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:

    def __init__(self):
        self.w=np.random.randn(2,3)#ガウス分布で(2,3)行列を初期化
    
    def predict(self,x):
        return np.dot(x,self.w)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)

        return loss


#試しに出力
net=SimpleNet()
print("w:"+str(net.w))
x=np.array([0.6,0.9])
p=net.predict(x)
print("出力:"+str(p))
t=np.array([0,0,1])
l=net.loss(x,t)
print("損失関数:"+str(l))  


def f(W):
    return net.loss(x,t)

dW=numerical_gradient(f,net.w)
print("勾配:"+str(dW))


_____________________________________________________________________________


#学習アルゴリズムの実装
#ステップ1:ミニバッチの選出
#ステップ2:勾配の算出
#ステップ3:パラメータの更新
#ステップ4:繰り返す

#2層(入力層-隠れ層-出力層)のニューラルネットワークで確率的勾配効果法(SGD)で学習を行う

#classでネットワークを定義する
import sys,os
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch04')  # カレントディレクトリをch04に変更
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class Twolayernet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #重みの初期化
        #np.random.randn(a,b):a×b行列のガウス分布に基づく乱数を生成
        self.prams={}
        self.prams['w1']=weight_init_std*np.random.randn(input_size,hidden_size)
        #バイアスは0で初期化
        self.prams['b1']=np.zeros(hidden_size)
        self.prams['w2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.prams['b2']=np.zeros(output_size)
    
    def predict(self,x):
        w1,w2=self.prams['w1'],self.prams['w2']
        b1,b2=self.prams['b1'],self.prams['b2']
        a1=np.dpt(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        y=softmax(a2)

        return y

    def loss(self,x,t):
        y=self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_w=lambda w:self.loss(x,t)
        grad={}
        grad['w1']=numerical_gradient(loss_w,self.prams['w1'])
        grad['b1']=numerical_gradient(loss_w,self.prams['b1'])
        grad['w2']=numerical_gradient(loss_w,self.prams['w2'])
        grad['b2']=numerical_gradient(loss_w,self.prams['b2'])

        return grad

    
net=TwoLayernet(input_size=784,hidden_size=100,output_size=10)
net.prams['w1'].shape

___________________________________________________________________________

#ミニバッチ学習の実装
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import numpy as np

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

train_loss_list=[]

#ハイパーパラメータ
iter_num=10000
train_size=x_train.shape[0]
batch_size=100
learnig_rate=0.1

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iter_num):
    #ミニバッチの取得
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #勾配の計算
    grad=network.numerical_gradient(x_batch,t_batch)

    #パラメータの更新
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learnig_rate*grad[key]
    
    #学習経過の記録
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

________________________________________________________________________________________

#テストデータで評価
import sys, os
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch04')  # カレントディレクトリをch04に変更
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_test,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
#1エポックあたりの繰り返し数
iter_per_epoch=max(train_size/batch_size,1)

#ハイパーパラメータ
iter_num=10000
batch_size=100
learnig_rate=0.1

network=TwoLayerNet(input_size=100,hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()






