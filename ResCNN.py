import tensorflow as tf
from tensorflow.python.training import moving_averages

'''残差网络运算函数'''

#调用使用的API/入口函数
def openFunction(inputImg):
    tensorImg = tf.Variable(inputImg,name='inputImg',dtype='float32')   #批量初始化

    return Rescnn().call(tensorImg)


#单个残差块
class ResidualBlock(tf.keras.Model):
    def __init__(self,outChannels,strides=1,use_1x1conv=False):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(outChannels,3,strides,'same')       #步幅为2的3x3卷积层1
        self.conv2 = tf.keras.layers.Conv2D(outChannels,3,padding='same')       #步幅为2的3x3卷积层2

        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(outChannels,1,strides,padding='same')   #用于改变大小和通道的1x1卷积层
        else:                                                                           #步幅为2
            self.conv3 = None

        self.bn1 = tf.keras.layers.BatchNormalization()         #批量归一化层1
        self.bn2 = tf.keras.layers.BatchNormalization()         #批量归一化层2

        self.relu = tf.keras.layers.ReLU()

    def call(self,tensorImg):
        Y = self.bn1(self.conv1(tensorImg))     #一次卷积，一次批量归一
        Y = self.relu(Y)                        #一次relu激活函数
        Y = self.bn2(self.conv2(Y))             #再一次卷积，再一次批量归一
        if self.conv3 is not None:
            tensorImg = self.conv3(tensorImg)
        Y += tensorImg                          #跨越卷积运算残差相加
        Y = self.relu(Y)
        return Y                     #返回结果


#34层残差网络实现
class Rescnn(tf.keras.Model):
    def __init__(self,isIraining=True):
        super().__init__()
        self.isTraining = isIraining

        self.conv7x7 = tf.keras.layers.Conv2D(64,7,2,'same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')
        #self.avgpool1 = tf.keras.layers.GlobalAvgPool2D()
        #self.dense1 = tf.keras.layers.Dense(units=10)
        self.relu = tf.keras.layers.ReLU()

        self.residualBlock64 = ResidualBlock(64)
        self.residualBlock128 = ResidualBlock(128)
        self.residualBlock128T = ResidualBlock(128,2,True)
        self.residualBlock256 = ResidualBlock(256)
        self.residualBlock256T = ResidualBlock(256,2,True)
        self.residualBlock512 = ResidualBlock(512)
        self.residualBlock512T = ResidualBlock(512,2,True)

    def call(self,Y):
        Y = self.bn1(self.conv7x7(Y))       #初始的7x7卷积和批量规范化/[batch,/2,/2,64]
        Y = self.relu(Y)                    #激活函数
        Y = self.maxpool1(Y)                #[batch,/2,/2,64]
                                                    #残差计算
        for i in range(0,2):                        #[batch,/2,/2,*2]
            Y = self.residualBlock64(Y)
        
        Y = self.residualBlock128T(Y)       #[batch,/2,/2,*2]
        for _ in range(0,2):
            Y = self.residualBlock128(Y)

        Y = self.residualBlock256T(Y)       #[batch,/2,/2,*2]
        for _ in range(0,4):
            Y = self.residualBlock256(Y)

        Y = self.residualBlock512T(Y)       #[batch,/2,/2,*2]
        for _ in range(0,2):
            Y = self.residualBlock512(Y)
        #print(Y.shape)

        return Y





    #以下为之前的代码，应该没用了
""" 
def initializer(name,shape,dtype='float32'):     #初始化函数
    return tf.Variable(tf.keras.initializers.HeNormal()(shape = shape),name=name,dtype=dtype)

    #卷积层
    def conv2d(self,tensorImg,numOutputs,kernelSize,stride = 1,scope = 'conv2d'):
        numInputs = tensorImg.get_shape()[-1]       #获得深度信息
        with tf.name_scope(scope):      #计算卷积，使用0填充
            kernel = self.initializer('kernal',shape=(kernelSize,kernelSize,numInputs,numOutputs))
            return tf.nn.conv2d(tensorImg,kernel,strides=[1,stride,stride,1],padding='SAME')
        

    #批量化归一层
    def batchNorm(self,tensorImg,decay = 0.999,epsilon = 1e-03,isTraining = True,scope = 'batchNorm'):
        inputImg_shape = tensorImg.get_shape()
        numInputs = tensorImg[-1]
        reduceDims = list(range(len(inputImg_shape) - 1))
        print(numInputs)
        print(tensorImg.get_shape())
        print('zzzzzzzzzzzzzzzzzzzzzzz')
        print([numInputs,])

        with tf.name_scope(scope):
            #创建初始为全零和全一的变量，用于训练
            beta = tf.Variable(tf.keras.initializers.Zeros()(shape=tensorImg.shape()),name='beta',dtype='float32')
            gamma = tf.Variable(tf.keras.initializers.Ones()(shape=tensorImg.shape()),name='gamma',dtype='float32')

            movingMean = self.initializer('movintMean',[numInputs,])
            movingVariance = self.initializer('movingVariance',[numInputs,])

        #用于滑动平均
        if isTraining:
            mean,variance = tf.nn.moments(tensorImg,reduceDims)
            update_move_mean = moving_averages.assign_moving_average(movingMean,mean,decay=decay)
            update_move_variance = moving_averages.assign_moving_average(movingVariance,variance,decay=decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_move_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_move_variance)
        else:
            mean,variance = movingMean,movingVariance
        return tf.nn.batch_normalization(tensorImg,mean,variance,beta,gamma,epsilon)
    

    #池化层
    def Pool(self,tensorImg,poolSize,stride,scope,type='max'):
        if type == 'max':
            with tf.name_scope(scope):
                return tf.nn.max_pool(tensorImg,[1,poolSize,poolSize,1],[1,stride,stride,1],padding='SAME')
        else:
            with tf.name_scope(scope):
                return tf.nn.avg_pool()  """
