#算術計算(python3系では小数表示)
7/5


#データ型
#整数int、小数float、文字列str
type(10)
type(2.788)
type("hellow")


#変数
x=91
print(x)


#リスト
a=[1,2,3,4,5]
a[0]
len(a)
#リストの最後から１つ目まで取得（スライス）
a[0:-1]


#ディクショナリ
me={'height':180,'weight':70}
me['height']


#関数
def hellow():
    print("hellow")
hellow() 

def hello(object):
    print("hello"+object+"!")
hello("TOTA")


#クラス
#class クラス名（引数）：
#   def __init__(self,引数):・・・コンストラクタ
#   def メソッド名(self、引数):・・・メソッド
#コンストラクタ・・・初期化を行うメソッド、
# 　　　　　　　　　インスタンス作成時に一度実行される

class Man:
    def __init__(self,name):
        self.name=name
        print('Initialized')
    
    def hello(self):
        print("Hello"+self.name+"!")
    
    def goodbye(self):
        print("Good-bye"+self.name+"!")

m=Man("TOTA")
m.hello()
m.goodbye()  


#numpy 
import numpy as np
x=np.array([1,2,3])
print(x)

A=np.array([[1,2],[3,4]])
print(A)
A.shape
A[1][0]


#Matplotlib
#sinカーブのプロット
import matplotlib.pyplot as plt
x=np.arange(0,6,0.1)
y=np.sin(x)
plt.plot(x,y)
plt.show()

#sin、cos,tanh
x1=np.arange(0,6,0.1)
y1=np.sin(x1)
y2=np.cos(x1)
y3=np.tanh(x1)
plt.plot(x1,y1,label="sin")
plt.plot(x1,y2,linestyle="--",label="cos")
plt.plot(x1,y3,color="red",label="tan")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin&cos&tanh")
plt.legend()
plt.show()

