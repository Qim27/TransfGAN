import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import einops
import ResCNN

'''transformerモデル'''

#API
def openFunction(lightList,darkList):
    tensorImg = tf.cast(tf.concat([lightList,darkList],1),'float16')
    #tensorImg = tf.Variable(inputImg,name='inputImg',dtype='float32')
    return visionTransformer(768,depth=6).call(tensorImg)



def openFunction_discriminator(lightList,darkList,generateImg):
    lightAddImg = tf.concat([lightList,generateImg],1)
    darkAddImg = tf.concat([darkList,generateImg],1)
    inputImg = tf.concat([lightAddImg,darkAddImg],2)            #[b,512,512,3]
    return Discriminator().call(inputImg)



#transformerブロック
class transformerBlock(tf.keras.Model):
    def __init__(self,dim,heads=8,dimHead=64):
        super().__init__()
        self.dim = dim
        self.heads =heads
        self.scale = dimHead ** -0.5        #除算用のスケールを計算
        Dropout = 0.01

        self.multiply3 = tf.keras.layers.Dense(self.dim * 3,'relu')     #3掛けてQKVに分割
        self.dense1 = tf.keras.layers.Dense(self.dim * 4,'relu')
        self.dense2 = tf.keras.layers.Dense(self.dim)
        self.dropout = tf.keras.layers.Dropout(Dropout)
        self.layerNorm = tf.keras.layers.LayerNormalization()           #レイヤー正規化
        self.softmax = tf.keras.layers.Activation(tf.nn.softmax)


    def call(self,tensorImg):
        Y = self.layerNorm(tensorImg)

        Y = self.attention(Y,self.heads)
        Y = Rearrange('b h n c -> b n (h c)')(Y)        #テンソルの次元を変更

        Y = Y + tensorImg          #残差接続
        Y1 = self.layerNorm(Y)
        return self.MLP(Y1) + Y


    #マルチヘッド自己注意機構
    def attention(self,tensorImg,heads=8):
        tensorImg = self.multiply3(tensorImg)
        tensorImg = Rearrange('b n (c h i) -> i b h n c',i=3,h=heads)(tensorImg)    #[b,256,768]->[3,b,8,256,96]

        Q = tensorImg[0]             #3分割してQKV行列に割り当て
        K = tensorImg[1]
        V = tensorImg[2]

        #attention関数
        Y = tf.matmul(Q,K,transpose_b=True,name='QKxT') * self.scale
        Y = tf.matmul(self.softmax(Y),V,name='softmaxV')
        return Y
    

    #多層パーセプトロン / 全結合層
    def MLP(self,tensorImg):
        Y = self.dense1(tensorImg)
        Y = self.dropout(Y)
        Y = self.dense2(Y)
        return self.dropout(Y)



#ViTメイン関数
class visionTransformer(tf.keras.Model):
    def __init__(self,dim,heads=8,depth=12,patchSize=16):
        super().__init__()
        self.dim = dim          #最後の次元サイズ
        self.patchSize = patchSize
        self.heads = heads
        self.depth = depth      #transformerブロックの繰り返し回数
        
        #以下の行でpatchに対して線形射影を行う
        self.linearProjection = tf.keras.layers.Dense(self.dim,'relu')       
        self.positionEmbedding = self.add_weight('positionEmbedding',[1,512,dim],'float32',                     #位置エンコーディング行列を追加
                                            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.transformerBlock = transformerBlock(dim,heads)
        
        
    def call(self,tensorImg):
        #[batch,256,256,3] -> [batch,256 / 16,256 / 16,3] -> [batch,256(patchNumber),16,16,3]
        # -> [batch,256,16 * 16 * 3] -> [batch,256,768]
        #入力画像を256個のpatchに分割。具体的な分割手順は上記の通り
        tensorImg = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize,p2=self.patchSize)(tensorImg)
        
        #Y = tf.concat([self.positionEmbedding,tensorImg],0,'addPositionEmbedding')
        Y = self.positionEmbedding + tensorImg          #[b,512,768]
        #print(tensorImg.shape)

        Y = self.linearProjection(Y)                    #線形変換後、transformerブロックに入力して計算
        for _ in range(self.depth):
            Y = self.transformerBlock(Y)

        return Y
    


class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.beforeCompute = beforeCompute()
        
        self.avgpool = tf.keras.layers.AveragePooling1D(4,4,'valid')
        #self.avgpool_2x = tf.keras.layers.AveragePooling1D(2,2,'valid')
        self.dence = tf.keras.layers.Dense(1)
        self.layerNorm = tf.keras.layers.LayerNormalization()
        self.tanh = tf.keras.layers.Activation(tf.nn.tanh)

        self.transformerBlock1 = transformerBlock(384)
        self.transformerBlock2 = transformerBlock(768)
        self.transformerBlock3 = transformerBlock(1536)
        #self.transformerBlock4 = transformerBlock(1536)
        
        self.positionEmbedding_4x = self.add_weight('positionEmbedding',[1,1024,384],'float32',                     #位置エンコーディング行列を追加
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.positionEmbedding_2x = self.add_weight('positionEmbedding',[1,256,384],'float32',
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.positionEmbedding = self.add_weight('positionEmbedding',[1,64,768],'float32',
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        
        self.classToken = self.add_weight(name='classToken',shape=[1,1,1536],dtype='float32',
                                          initializer=tf.keras.initializers.GlorotUniform(),trainable=True)

    def call(self,inputImg):
        tensorImg_4x,tensorImg_2x,tensorImg = self.beforeCompute(inputImg)      #[b,1024,384] / [b,256,384] / [b,64,768]
        # tensorImg_4x = tensorImg_4x + self.positionEmbedding_4x
        # tensorImg_2x = tensorImg_2x + self.positionEmbedding_2x
        # tensorImg = tensorImg + self.positionEmbedding

        for _ in range(3):
            Y = self.transformerBlock1(tensorImg_4x)
        Y = self.avgpool(Y)                             #->[b,256,384]

        Y = tf.concat([Y,tensorImg_2x],-1)              #->[b,256,768]
        for _ in range(3): 
            Y = self.transformerBlock2(Y)
        Y = self.avgpool(Y)                             #->[b,64,768]

        Y = tf.concat([Y,tensorImg],-1)                 #->[b,64,1536]
        b,_,c = Y.shape
        for _ in range(3):
            Y = self.transformerBlock3(Y)
        
        classToken = tf.broadcast_to(self.classToken,[b,1,c],'classToken')
        Y = tf.concat([Y,classToken],1)                 #class tokenを追加して特徴抽出用の変数とする
        Y = self.transformerBlock3(Y)
        Y = self.layerNorm(Y)
            
        classToken = self.dence(Y[: , 0])               #[b,1536]        
        
        #print(classToken[: , 0])

        return classToken[: , 0]


class beforeCompute(tf.keras.layers.Layer):                 #入力データの前処理
    def __init__(self,channelSize=96,patchSize=8):
        super().__init__()
        self.channelSize = channelSize
        self.patchSize = patchSize
        self.patchSize1_2 = int(patchSize / 4)
        self.conv1 = tf.keras.layers.Conv2D(self.channelSize,self.patchSize,self.patchSize,'valid')
        self.conv2 = tf.keras.layers.Conv2D(self.channelSize,self.patchSize * 2,self.patchSize * 2,'valid')
        self.conv3 = tf.keras.layers.Conv2D(self.channelSize * 2,self.patchSize * 4,self.patchSize * 4,'valid')

    def build(self,_):
        None

    def call(self,inputImg):
        inputImg_4x = self.conv1(inputImg)              #ダウンサンプリングして[b,64,64,96]にする
        inputImg_2x = self.conv2(inputImg)              #ダウンサンプリングして[b,32,32,96]にする
        inputImg = self.conv3(inputImg)                 #ダウンサンプリングして[b,16,16,192]にする
        tensorImg_4x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg_4x)                       #->[b,1024,384]
        tensorImg_2x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg_2x)                       #->[b,256,384]
        tensorImg = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg)                             #->[b,64,768]

        return [tensorImg_4x,tensorImg_2x,tensorImg]


