#Affineレイヤ
import numpy as np
X=np.random.rand(2)
#1行2列の行列
print(X)

W=np.random.rand(2,3)
#2行3列の行列
print(W)

B=np.random.rand(3)
#1行3列の行列
print(B)

Y=np.dot(X,W)+B
print(Y)

#softmax-with-loss

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    exp_sum_a=np.sum(exp_a)
    y=exp_a/exp_sum_a
    
    return y

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    
    batch_size=y.shape[0]
    return -np.sun(t*np.log(y+1e-7))/batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)

        return self.loss
    
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self,y-self.t)/batch_size

        return dx

#誤差逆伝搬法の実装
#全体図
#1.ミニバッチを取り出す
#2.勾配の算出
#3.パラメータの更新
#4.繰り返す

import sys,os 
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch04')  
sys.path.append(os.pardir)
import numpy as np
#layer.pyファイルには種々のclassが収納されている
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        #パラメータの初期化
        self.params['w1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['w2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self,params['b2']=np.zeros(output_size)

        #レイヤの生成
        #順番付きのディクショナリ:OrderedDict
        self.layers=OrderedDict()
        #class Affineをインスタンス化(第一引数(__init__での）はw、b)
        self.layers['Affine1']=Affine(self.params['w1'],self.params['b1'])
        #class Reluをインスタンス化（第一引数なし）
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['w2'],self.params['b2'])
        self.lastLayer=SoftmaxWithLoss()

    def predict(self,x):
        #layerはforのための変数
        for layer in self.layers.values():
            #上記の種々のクラスそれぞれについて、メソッドforwardで順方向の伝搬を行う
            x=layer.forward(x)

            return x

    def loss(self,x,t):
        y=self.predict(x)

        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim!=1:t=np.argmax(t,axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])

        return accuracy

#数値微分による勾配算出(前章と同じものであり低速)
def numerical_gradient(self,x,t):
    loss_w=lambda q:self.loss(x,t)

    grads={}
    grads['w1']=numerical_gradient(loss_w,self.params['w1'])
    grads['b1']=numerical_gradient(loss_w,self.params['b1'])
    grads['w2']=numerical_gradient(loss_w,self.params['w2'])
    grads['b2']=numerical_gradient(loss_w,self.params['b2'])

    return grads

#誤差逆伝播による勾配算出(本性の内容であり高速)
def gradient(self,x,t):
    #forward
    self.loss(x,t)
    #backward
    dout=1
    dout=self.lastLayer.backward(dout)

    layers=list(self.layers.value())
    layers.reverse()
    for layer in layers:
        dout=layers.backward(dout)

    #設定
    grads = {}
    grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
    grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

    return grads


#誤差逆伝播法の勾配確認
#数値微分での勾配結果と、誤差逆伝搬法による勾配を比較評価する：勾配確認

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

#３個の訓練データをバッチとして用いて勾配評価する
x_batch=x_train[:3]
t_batch=t_train[:3]

#数値微分
grad_numerical=network.numerical_gradient(x_batch,t_batch)
#誤差逆伝播
grad_backprop=network.gradient(x_batch,t_batch)

#辞書型dictにおいて{key:value}の対応関係である
for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))





#誤差逆伝播を使った学習
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)


#ハイパーパラメータ
iter_num=10000
train_size=x_train.shape[0]

batch_size=100
learnig_rate=0.1

#評価用の指標のリストを生成
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iter_per_epoch=max(1,train_size/batch_size)

for i in range(iter_num):
    batch_mask=np.random.choice(train_size,batch_size)
    #batch_maskに対応するインデックスをバッチデータとして使用する
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    #誤差逆伝播による勾配計算
    grad=network.gradient(x_batch,t_batch)

    #更新
    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learnig_rate*grad[key]

    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    #epoch数の時の試行について
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train,t_train)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)



