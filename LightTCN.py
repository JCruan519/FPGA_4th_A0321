from keras.models import Sequential
from keras import layers,regularizers
from keras.models import Model
from keras.layers import add,Input,Conv1D,Activation,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization as BN
import keras
import keras.backend as K


def LightBlock(x,f,k,dilation_rate):
    '''
    函数作用：轻量化模块(WITHOUT CHANNEL SHUFFLE)，将原TCN网络中的一维卷积形式变成一维DW+Point卷积，减小参数量
    输入：x-->特征向量；f-->卷积核个数；k-->卷积核大小；dilation_rate：扩张系数
    输出：处理后的特征向量
    '''
    r=Conv1D(filters=1,kernel_size=k,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=Conv1D(filters=f,kernel_size=1,activation='relu')(r)
    
    if x.shape[-1]==f:
        shortcut=x
    else:
        shortcut=Conv1D(filters=f,kernel_size=k,padding='same')(x)  #shortcut（捷径）
    o=add([r,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o
 
#序列模型
def LightTCN():
    '''
    函数作用：将LightBlock串接，组成LightTCN网络
    '''
    inputs=Input(shape=(256,1)) ##信号长度
#     x=ResBlock(inputs,f=64,k=5,dilation_rate=1)
#     x=ResBlock(x,f=32,k=5,dilation_rate=2)
#     x=ResBlock(x,f=16,k=3,dilation_rate=4)
#     x=ResBlock(x,filters=8,kernel_size=3,dilation_rate=8)
#     x=ResBlock(x,filters=8,kernel_size=3,dilation_rate=8)
#     x=ResBlock(inputs,f=256,k=7,dilation_rate=1)
#     x=ResBlock(x,f=128,k=5,dilation_rate=2)
#     x=ResBlock(x,f=64,k=5,dilation_rate=4)
    x=LightBlock(inputs,f=32,k=3,dilation_rate=1)
    x=LightBlock(x,f=16,k=3,dilation_rate=2)
    x=LightBlock(x,f=8,k=3,dilation_rate=4)
    
    x=Flatten()(x)
    x = BN()(x)
    
    x=Dense(6,activation='softmax')(x) ###分类个数
    model=Model(inputs,x)
    
    return model