#ニューラルネットワーク
#ステップ関数の実装
#実数引数のステップ関数(numpy配列に対応していない))
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

#numpy 配列対応のステップ関数
#np.astype():データ型の変換
import numpy as np 
import matplotlib.pylab as plt

def step_function2(x):
    return np.array(x>0,dtype=np.int)

x=np.arange(-5,5,0.1)
y=step_function2(x)
plt.plot(x,y)
plt.show()

#シグモイド関数
#シグモイド
def sigmoid(x):
    return 1/(1+np.exp(-x))

#numpy対応のシグモイド
x=np.arange(-5,5,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.show()


#Relu関数
def relu(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.01)
y=relu(x)
plt.plot(x,y)
plt.show()


#ニューラルネットワークの行列の積
X=np.array([1,2])
X.shape
W=np.array([[1,3,5],[2,4,6]])
print(W)

#x1        y1
#          y2
#x2        y3
#の形のニューラルネットを考える(ノード間は全結合、バイアスは考慮しないとする)
Y=np.dot(X,W)
print(Y)


#3層ニューラルネットワークの実装
#x1 @   @  y1
#   @
#x2 @   @  y2
#の形を考える
#重みの記号の意味
#Wji:前層のi番目のニューロンから次層j番目のニューロンに対する重み

#シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#辞書型関数でネットワークのパラメータを定義
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])

    return network

#行列計算
def forward(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=a3

    return y

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
print(y)

#出力層の設計
#分類問題で用いるソフトマックス関数
#y(k)=exp(ak)/Σexo(a)

#ソフトマックス関数の実現
a=np.array([0.3,2.9,4.0])
exp_a=np.exp(a)
sum_exp_a=np.sum(exp_a)
y=exp_a/sum_exp_a
print(y)

#関数として定義
def softmax_1(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y



 #オーバーフロー対策
 #分母分子に定数Cをかける
 #正しく計算されない
a=np.array([1010,1000,990])
softmax(a)

#softmax改
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    exp_sum_a=np.sum(exp_a)
    y=exp_a/exp_sum_a

    return y

a=np.array([0.3,2.9,4.0])
y=softmax(a)
print(y)


#出力層のニューロンの数
#学習済みのパラメータを用いて手書き文字認識
#28*28の0~255の1チャンネル画像にラベル
import sys, os
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch03')  # カレントディレクトリをch03に変更
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

#(訓練画像、訓練ラベル)、(テスト画像、テストラベル)
#flatten:一次元配列に変換するか
#normalize:ピクセルを正規化するか
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)

#mnist画像の表示
import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
img=x_train[0]
label=t_train[0]
print(label)
img=img.reshape(28,28)
img_show(img)

#入力層が784、出力層が10ラベルのニューラルネットワークを設計
#隠れ層が二層（50,100)

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#データの前処理を行いaccurucyを改善する
#バッチ処理
#数枚まとめて入力する
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

#ここからがバッチ処理！！
x, t = get_data()
network = init_network()
batch_size=10

accuracy_cnt = 0
#range(a,b,c):aからc-1までcおきのリストを作成
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p= np.argmax(y_batch,axis=1) #axis:次元（ここでは1行)に対して最大値を求める)
    accuracy_cnt +=np.sum(p==t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
