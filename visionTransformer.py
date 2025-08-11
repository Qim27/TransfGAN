import tensorflow as tf
from einops.layers.tensorflow import Rearrange
import einops
import ResCNN

'''transformer部分/transformerモデル'''

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



#transformer块/transformerブロック
class transformerBlock(tf.keras.Model):
    def __init__(self,dim,heads=8,dimHead=64):
        super().__init__()
        self.dim = dim
        self.heads =heads
        self.scale = dimHead ** -0.5        #计算要除的
        Dropout = 0.01

        self.multiply3 = tf.keras.layers.Dense(self.dim * 3,'relu')     #乘三分给QKV
        self.dense1 = tf.keras.layers.Dense(self.dim * 4,'relu')
        self.dense2 = tf.keras.layers.Dense(self.dim)
        self.dropout = tf.keras.layers.Dropout(Dropout)
        self.layerNorm = tf.keras.layers.LayerNormalization()           #批量层规范化
        self.softmax = tf.keras.layers.Activation(tf.nn.softmax)


    def call(self,tensorImg):
        Y = self.layerNorm(tensorImg)

        Y = self.attention(Y,self.heads)
        Y = Rearrange('b h n c -> b n (h c)')(Y)        #改变张量维度

        Y = Y + tensorImg          #残差连接
        Y1 = self.layerNorm(Y)
        return self.MLP(Y1) + Y


    #多头自注意力部分
    def attention(self,tensorImg,heads=8):
        tensorImg = self.multiply3(tensorImg)
        tensorImg = Rearrange('b n (c h i) -> i b h n c',i=3,h=heads)(tensorImg)    #[b,256,768]->[3,b,8,256,96]

        Q = tensorImg[0]             #分三份给QKV矩阵
        K = tensorImg[1]
        V = tensorImg[2]

        #attention函数
        Y = tf.matmul(Q,K,transpose_b=True,name='QKxT') * self.scale
        Y = tf.matmul(self.softmax(Y),V,name='softmaxV')
        return Y
    

    #多层感知机/全连接层
    def MLP(self,tensorImg):
        Y = self.dense1(tensorImg)
        Y = self.dropout(Y)
        Y = self.dense2(Y)
        return self.dropout(Y)



#ViT主函数/ViT関数
class visionTransformer(tf.keras.Model):
    def __init__(self,dim,heads=8,depth=12,patchSize=16):
        super().__init__()
        self.dim = dim          #最后一维的大小
        self.patchSize = patchSize
        self.heads = heads
        self.depth = depth      #transformer块的循环次数
        
        #下句对patch进行linear Projection
        self.linearProjection = tf.keras.layers.Dense(self.dim,'relu')       
        self.positionEmbedding = self.add_weight('positionEmbedding',[1,512,dim],'float32',                     #添加位置编码矩阵
                                            initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.transformerBlock = transformerBlock(dim,heads)
        
        
    def call(self,tensorImg):
        #[batch,256,256,3] -> [batch,256 / 16,256 / 16,3] -> [batch,256(patchNumber),16,16,3]
        # -> [batch,256,16 * 16 * 3] -> [batch,256,768]
        #输入图片切分成256个patch，具体切分步骤如上
        tensorImg = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize,p2=self.patchSize)(tensorImg)
        '''
        用不到而且报错，所以注释掉了
        #以下是添加classToken参数语句
        batchSize = list(tensorImg.shape)[0]
        classToken = self.add_weight('classToken',[1,1,self.dim],'float32',
                                     initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        classToken = tf.broadcast_to(classToken,[batchSize,1,self.dim])
        tensorImg = tf.concat([classToken,tensorImg],1,'addClassToken')     #[b,256,768]->[b,257,768]
        '''
        
        #Y = tf.concat([self.positionEmbedding,tensorImg],0,'addPositionEmbedding')
        Y = self.positionEmbedding + tensorImg          #[b,512,768]
        #print(tensorImg.shape)

        Y = self.linearProjection(Y)                    #线性加权后输入到transformer块中计算
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
        
        self.positionEmbedding_4x = self.add_weight('positionEmbedding',[1,1024,384],'float32',                     #添加位置编码矩阵
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.positionEmbedding_2x = self.add_weight('positionEmbedding',[1,256,384],'float32',
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        self.positionEmbedding = self.add_weight('positionEmbedding',[1,64,768],'float32',
                                           initializer=tf.keras.initializers.GlorotUniform(),trainable=True)
        
        self.classToken = self.add_weight(name='classToken',shape=[1,1,1536],dtype='float32',
                                          initializer=tf.keras.initializers.GlorotUniform(),trainable=True)

    def call(self,inputImg):
        #print('生成器运行结束，鉴别器开始运行')
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
        Y = tf.concat([Y,classToken],1)                 #添加class token特征提取变量
        Y = self.transformerBlock3(Y)
        Y = self.layerNorm(Y)
            
        classToken = self.dence(Y[: , 0])               #[b,1536]        
        
        #print(classToken[: , 0])

        return classToken[: , 0]

    '''暂时没用到
    def changeTensor_to_transformer(self,Y,patchSize=4):
        _,h,w,c = Y.shape
        h = int(h / patchSize)
        c = c * patchSize * patchSize
        Y = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=patchSize,p2=patchSize)(Y)        #[b,(h/4*w/4),c*16]
        Y = transformerBlock(c).call(Y)
        return Rearrange('b (n n2) (c p1 p2) -> b (n p1) (n2 p2) c',n2=h,p1=patchSize,p2=patchSize)(Y)      #上方逆过程
    '''


class beforeCompute(tf.keras.layers.Layer):                 #对输入数据进行预先处理
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
        inputImg_4x = self.conv1(inputImg)              #降采样至[b,64,64,96]
        inputImg_2x = self.conv2(inputImg)              #降采样至[b,32,32,96]
        inputImg = self.conv3(inputImg)                 #降采样至[b,16,16,192]
        tensorImg_4x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg_4x)                       #->[b,1024,384]
        tensorImg_2x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg_2x)                       #->[b,256,384]
        tensorImg = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)',p1=self.patchSize1_2,p2=self.patchSize1_2)(inputImg)                             #->[b,64,768]

        return [tensorImg_4x,tensorImg_2x,tensorImg]

