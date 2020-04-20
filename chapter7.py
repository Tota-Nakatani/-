#４次元データ
#(batch_num,channnel.height,width)

#conv,poolの実装
#データの形状が(10,1,28,28)
#28*28のサイズ、１チャンネルのデータが10個まとまった(batch)

#4次元配列の定義
import numpy as np
x=np.random.randn(10,1,28,28)
x.shape

#im2colによる展開
#im2colとはフィルタにとって都合の良いように入力データを展開する関数
#多次元配列(4次元)同士の演算を行列積演算に

#convの実装
#im2colは中身を気にせず利用
#以下の要素を持つ
#1.input_data
#2.filter_h
#3.filter_w
#4.stride
#5.pad

#im2colの実装
import sys,os
os.chdir('/Users/nakatanitota/deep-learning-from-scratch-master/ch04')  
sys.path.append(os.pardir)
from common.util import im2col

x1=np.random.rand(1,3,7,7)
col1=im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

#clas conv
class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad
    
    def forward(self,x):
        #フィルタのパラメータ
        FN,C,FH,FW=self.w.shape
        #入力のパラメータ
        N,C,H,W=self.x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        
        col=in2col(x,FH,FW,self.stride,self.pad)
        #フィルタも合わせて変形(一次元に)
        col_W=self.W.reshape(FN,-1).T
        out=np.dot(col,col_w)+self.b

        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        return out


#poolingレイヤの実装
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

        def forward(self,x):
            N,C,H,W=x.shape
            out_h=int(1+(H-self.pool_h)/self.stride)
            out_w=int(1+(W-self.pool_w)/self.stride)

            #展開
            col=im2col(x,self.pool_w,self.pool_w,self.stride,self.pad)
            col=col.reshape(-1,self.pool_h*self.pool_w)

            #最大値
            out=np.max(col,axis=1)
            #整形
            out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

            return out


#CNNの実装
#conv-relu-pool-affine-relu-affine-softmax
class SimpleConvNet:
    def __init__(self,input_dim=(1,28,28),conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},hidden_size=100,output_size=10,weight_init_std=0.01):
        filter_num=conv_param['filter_num']
        fulter