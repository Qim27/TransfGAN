import tensorflow as tf
import numpy as np
from einops.layers.tensorflow import Rearrange

import ResCNN

'''特徴強化および画像復元関数'''

def openFunction(lightRes,darkRes,ViTImg):

    Y = enhanceBlock(512).call(lightRes,darkRes)
    _,_,_,c = Y.shape
    return decoder(c).call(Y,ViTImg)



class enhanceBlock(tf.keras.Model):
    def __init__(self,inChannel):
        super().__init__()
        self.conv3 = tf.keras.layers.Conv2D(inChannel,1,[2,1])
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.01)

        self.resBlock1 = ResCNN.ResidualBlock(512)
        self.resBlock2 = ResCNN.ResidualBlock(512)


    def call(self,lightRes,darkRes):        #横方向に連結した後、2つの畳み込みブロックを接続
        #Y = tf.concat([lightRes,darkRes],1)
        #Y = self.conv3(Y)
        #Y = self.relu(Y)
        Y = (lightRes + darkRes) / 2

        Y = self.resBlock1(Y)               #残差ブロックを用いた特徴強化
        Y = self.resBlock2(Y)
        Y = self.dropout(Y)
        return Y                            #->[b,8,8,512]
    

#下の temp 関数に置き換えられている
class decoder(tf.keras.Model):
    def __init__(self,inChannel):
        super().__init__()
        channel1 = int(inChannel / 4)
        channel2 = int(channel1 / 4)

        self.conv1 = tf.keras.layers.Conv2D(channel1,3,padding='same')
        self.conv2 = tf.keras.layers.Conv2D(channel2,3,padding='same')
        self.conv3 = tf.keras.layers.Conv2D(3,1,padding='same')

        self.deconv1 = tf.keras.layers.Conv2DTranspose(channel1 * 2,7,8)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(channel2 * 2,4,4)

        self.relu = tf.keras.layers.ReLU()
        self.tanh = tf.keras.layers.Activation(tf.nn.tanh)
        self.perconv = tf.keras.layers.Conv2D(inChannel,1,[3,1])

        self.CVbE = CVbE()


    def call(self,tensorImg,ViTImg):                        #2つの畳み込みブロックと1×1畳み込み層
        ViTImg = self.CVbE(ViTImg)
        Y = (tensorImg + ViTImg) / 2

        Y = self.conv1(self.deconv1(Y))                     #逆畳み込みで空間サイズを拡大し、畳み込みで次元を削減
        Y = self.relu(Y)

        Y = self.conv2(self.deconv2(Y))
        Y = self.relu(Y)
        Y = self.conv3(Y)

        return Y


class CVbE(tf.keras.layers.Layer):                      #上記処理を実装したカスタムレイヤー
    def __init__(self,channelSize=512,patchSize=4):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(128,8,8,'valid')
        self.conv2 = tf.keras.layers.Conv2D(channelSize,patchSize,patchSize,'valid')
        self.conv3 = tf.keras.layers.Conv2D(channelSize,1,[2,1])
        self.relu = tf.keras.layers.ReLU()

    def build(self,_):
        None

    def call(self,ViTImg):
        Y = Rearrange('b (n p) (c p1 p2) -> b (n p) (p1 p2) c',p1=16,p2=16,p=16)(ViTImg)

        Y = self.conv2(self.conv1(Y))
        return self.relu(self.conv3(Y))




class decoder_temp(tf.keras.Model):
    def __init__(self,inChannel):
        super().__init__()
        channel1 = int(inChannel / 4)
        channel2 = int(channel1 / 4)

        self.conv1 = tf.keras.layers.Conv2D(channel1,3,padding='same')
        self.conv2 = tf.keras.layers.Conv2D(channel2,3,padding='same')
        self.conv3 = tf.keras.layers.Conv2D(3,1,padding='same')

        self.deconv1 = tf.keras.layers.Conv2DTranspose(channel1 * 2,7,8)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(channel2 * 2,4,4)

        self.relu = tf.keras.layers.ReLU()
        #self.tanh = tf.keras.layers.Activation(tf.nn.tanh)
        self.perconv = tf.keras.layers.Conv2D(inChannel,1,[3,1])


    def call(self,Y):                       #2つの畳み込みブロックと1×1畳み込み層
        Y = self.conv1(self.deconv1(Y))               #逆畳み込みで空間サイズを拡大し、畳み込みで次元を削減
        Y = self.relu(Y)

        Y = self.conv2(self.deconv2(Y))
        
        Y = self.relu(Y)
        Y = self.conv3(Y)

        return Y
        
    

    

