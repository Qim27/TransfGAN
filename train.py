import inputImage
import ResCNN
import visionTransformer
import EnhanceAndDecoder
import evaluation

import numpy
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



'''训练用'''
class allModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model_res = ResCNN.Rescnn()                                #读取模型
        self.model_res2 = ResCNN.Rescnn()
        #self.model_ViT = visionTransformer.visionTransformer(768)
        self.model_entance = EnhanceAndDecoder.enhanceBlock(512)
        #self.model_decoder = EnhanceAndDecoder.decoder(512)
        self.model_decoder_temp = EnhanceAndDecoder.decoder_temp(512)

    def call(self,inputList):
        inputList_lightS,inputList_darkS = inputList
        tensor_resLight = self.model_res(inputList_lightS)
        tensor_resDark = self.model_res2(inputList_darkS)

        #tensor_ViT = self.model_ViT(tf.cast(tf.concat([inputList_lightS,inputList_darkS],1),'float32'))
        #Y = self.model_decoder(self.model_entance(tensor_resLight,tensor_resDark),tensor_ViT)
        Y = self.model_decoder_temp(self.model_entance(tensor_resLight,tensor_resDark))
        #Y = self.model_decoder_temp(tensor_resLight,tensor_resDark)
        return Y
    


#超参数定义
learnRate = 0.001
numberEpochs = 20
allNumber = 20
batchSize_s = 5
testImang_filerPath = './Dataset_Part1/'
modelSave_path = './model/'


class training():
    def __init__(self):
        print('开始训练')
        self.model = allModel()                                     #读取模型
        self.discriminator = visionTransformer.Discriminator()
        #self.cosineDecay_lr = tf.keras.optimizers.schedules.CosineDecay(learnRate,3000)
        self.exponentialDecay_lr = tf.keras.optimizers.schedules.ExponentialDecay(learnRate,20,0.999)
        #self.generatorOptimizer = tf.keras.optimizers.Adam(self.cosineDecay_lr)
        self.discriminatorOptimizer = tf.keras.optimizers.SGD(0.0005)

        '''
        self.model.trainable = False            #训练判别器模型
        inputListL = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        inputListD = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        inputListN = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)

        generatedImage = self.model([inputListL,inputListD])
        falseOut = self.discriminator(generatedImage)
        realOut = self.discriminator(inputListN)
        mixWeight = tf.random.uniform(batchSize_s)
        mixOut = (realOut * mixWeight) + ((1 - mixWeight) * falseOut)
        mixOut_Out = self.discriminator(mixOut)
        self.adversarialModel = tf.keras.Model(inputs=[inputListL,inputListD,inputListN],outputs=[realOut,falseOut,mixOut_Out])
        self.adversarialModel.compile(loss=[lossFunction_wasserstein(),lossFunction_wasserstein(),lossFunction_gradientPenalty()],
                                      optimizer=tf.keras.optimizers.Adam(learnRate),loss_weights=[1,1,10])
        
        self.model.trainable = True
        self.discriminator.trainable = False
        inputListL = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        inputListD = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        generatedImage = self.model([inputListL,inputListD])
        falseOut = self.discriminator(generatedImage)
        self.generatorModel = tf.keras.Model(inputs=[inputListL,inputListD],outputs=falseOut)
        self.generatorModel.compile(loss=lossFunction_wasserstein(),optimizer=self.discriminatorOptimizer)
        '''
        self.discriminator.compile(optimizer=self.discriminatorOptimizer,loss='mean_squared_error',metrics=['Accuracy','binary_crossentropy'])
        self.adversarialModel = self.adversarial()        
            


    def call(self):         #开始训练
        inputList_light_trainData,inputList_dark_trainData,inputList_normal_trainData, \
            inputList_light_testData,inputList_dark_testData,inputList_normal_testData = self.inputImage_all(testImang_filerPath)

        endTime,allTime,nowTime = [0,0,0]
        batchSum = int(allNumber / batchSize_s)


        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.exponentialDecay_lr),loss=[lossFunction_meanSquared(),lossFunction_ssim()],metrics=['Accuracy','binary_crossentropy'],loss_weights=[0.1,20])
        history = self.model.fit([inputList_light_trainData,inputList_dark_trainData],inputList_normal_trainData,
                                 batchSize_s,numberEpochs,validation_data=([inputList_light_testData,inputList_dark_testData],inputList_normal_testData),shuffle=False)
        self.model.save(f'{modelSave_path}model22-GOnonD.tf')
        outImage_train = self.model([inputList_light_trainData[:20],inputList_dark_trainData[:20]])
        outImage_test = self.model([inputList_light_testData[:20],inputList_dark_testData[:20]])
        inputImage.saveImage(inputList_light_trainData[:20],'./outImages/light/')
        inputImage.saveImage(inputList_dark_trainData[:20],'./outImages/dark/')
        inputImage.saveImage_tensorImg(outImage_train)
        inputImage.saveImage_tensorImg(outImage_test,'./outImages/test/')
        inputImage.saveImage(inputList_normal_trainData[:20],'./outImages/normal/')
        evaluation.test().call()
        print('training is over')
        return
        
        
        for i in range(numberEpochs):           #训练循环
            
            allTime = endTime - nowTime + allTime
            print(f'====== epoch:{i+1}/{numberEpochs}. time spent:{(endTime-nowTime):.3f}, total time:{allTime:.3f}. ======')             #输出基本信息
            nowTime = tf.timestamp()
            for j in range(batchSum):
                self.train_step([inputList_light_trainData[5*j:5*(j+1)],inputList_dark_trainData[5*j:5*(j+1)]],inputList_normal_trainData[5*j:5*(j+1)])
                
            #验证部分
            generatedImage_validation = self.model([inputList_light_testData,inputList_dark_testData])
            GLoss = lossFunction_generator().call(inputList_normal_testData,generatedImage_validation,[0.1,5])
            DLoss = lossFunction_crossEntropy().call(self.discriminator(generatedImage_validation),self.discriminator(inputList_normal_testData))
            print(f'validation part : Gloss: {GLoss[0]} ,(MS:{GLoss[1]},ssim:{GLoss[2]}), Dloss: {DLoss} \n')
            endTime = tf.timestamp()

        try:
            self.modelSave(20,'tf')                      #保存
        except:
            print('save error')
        else:
            print('model is saved')
       
        '''
        #使用keras提供的方法训练模型
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learnRate),loss=[lossFunction_meanSquared(),lossFunction_ssim()],
                           metrics=['Accuracy','mean_squared_error'],loss_weights=[0.5,20])
        
        history = self.model.fit([inputList_light_trainData,inputList_dark_trainData],inputList_normal_trainData,
                                 batchSize_s,numberEpochs,validation_data=([inputList_light_testData,inputList_dark_testData],inputList_normal_testData),shuffle=False)
        '''    
        
        outImage_train = self.model([inputList_light_trainData[:20],inputList_dark_trainData[:20]])
        outImage_test = self.model([inputList_light_testData[:20],inputList_dark_testData[:20]])
        inputImage.saveImage(inputList_light_trainData[:20],'./outImages/light/')
        inputImage.saveImage(inputList_dark_trainData[:20],'./outImages/dark/')
        inputImage.saveImage_tensorImg(outImage_train)
        inputImage.saveImage_tensorImg(outImage_test,'./outImages/test/')
        inputImage.saveImage(inputList_normal_trainData[:20],'./outImages/normal/')
        evaluation.test().call()
        print('training is over')



    def train_step(self,inputList,trueImage):              #定义训练每步的内容
        genertedImage = self.model(inputList)

        discriminatorLoss1 = self.discriminator.train_on_batch(trueImage,tf.ones(batchSize_s,'float32'))                        #先训练鉴别器
        discriminatorLoss2 = self.discriminator.train_on_batch(genertedImage,tf.zeros(batchSize_s,'float32'))
        discriminatorLoss = (discriminatorLoss1[0] + discriminatorLoss2[0]) / 2

        generatorLoss = self.adversarialModel.train_on_batch(inputList,[trueImage,trueImage,tf.ones(batchSize_s,'float32')])    #后训练生成器
        #输出结果
        print(f'Gloss: {generatorLoss[0]:.9f}, (MSLoss:{generatorLoss[1]:.4f}, ssimLoss:{generatorLoss[2]:.9f}, DLoss:{generatorLoss[3]:.7f}),Accuracy: {generatorLoss[4]:.9f}, Dloss: {discriminatorLoss:.7f} ')
        return 

    
    def adversarial(self):      #训练生成器用模型
        inputList1 = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        inputList2 = tf.keras.layers.Input(shape=(256,256,3),batch_size=batchSize_s)
        generatorOut = self.model([inputList1,inputList2])
        self.discriminator.trainable = False
        falseOut = self.discriminator(generatorOut)
        model = tf.keras.Model(inputs=[inputList1,inputList2],outputs=[generatorOut,generatorOut,falseOut])

        model.compile(optimizer=tf.keras.optimizers.Adam(self.exponentialDecay_lr),loss=[lossFunction_meanSquared(),lossFunction_ssim(),'mean_squared_error'],
                      metrics=['Accuracy','binary_crossentropy'],loss_weights=[0.1,20,0.05])
        
        return model



    #读取数据集
    def inputImage_all(self,testImang_filerPath,allNumber=allNumber):
        inputList_light = []
        inputList_dark = []
        inputList_normal = []
        i = 0
        for i in range(10):
            imgPath_light = testImang_filerPath + f'{i+1}/4.JPG'
            imgPath_dark = testImang_filerPath + f'{i+1}/3.JPG'
            imgPath_normal = testImang_filerPath + f'{i+1}/8.JPG'
            
            inputList_light.extend(inputImage.inImage(imgPath_light))           #读取图片
            inputList_dark.extend(inputImage.inImage(imgPath_dark))
            inputList_normal.extend(inputImage.inImage(imgPath_normal))

            if(len(inputList_light) > allNumber):                               #到数量就退出循环
                break
            

        state = numpy.random.get_state()                #按相同顺序打乱三个数据集
        numpy.random.shuffle(inputList_light)
        numpy.random.set_state(state)
        numpy.random.shuffle(inputList_dark)
        numpy.random.set_state(state)
        numpy.random.shuffle(inputList_normal)

        allSize = len(inputList_light)                  #分成训练用和测试用数据集
        trainSize = int(allSize * 0.9)

        inputList_light_trainData = numpy.array(inputList_light[ : trainSize],'float32')
        inputList_dark_trainData = numpy.array(inputList_dark[ : trainSize],'float32')
        inputList_normal_trainData = numpy.array(inputList_normal[ : trainSize],'float32')

        inputList_light_testData = numpy.array(inputList_light[trainSize : ],'float32')
        inputList_dark_testData = numpy.array(inputList_dark[trainSize : ],'float32')
        inputList_normal_testData = numpy.array(inputList_normal[trainSize : ],'float32')
        
        # imgIn_light = []
        # imgIn_dark = []
        # imgIn_normal = []
        # for i in range(allNumber):
        #     imgIn_light.extend(inputImage.inImage(f'./Dataset_Part1/light/{i+1}.jpg'))
        #     imgIn_dark.extend(inputImage.inImage(f'./Dataset_Part1/dark/{i+1}.jpg'))
        #     imgIn_normal.extend(inputImage.inImage(f'./Dataset_Part1/normal/{i+1}.jpg'))

        # inputList_light_trainData = numpy.array(imgIn_light,'float32')
        # inputList_dark_trainData = numpy.array(imgIn_dark,'float32')
        # inputList_normal_trainData = numpy.array(imgIn_normal,'float32')

        '''
        #变为dataset对象，并转为float16格式
        inputList_light_trainData = tf.data.Dataset.from_tensor_slices(inputList_light_trainData,'inputList_light_trainData').map(lambda a : tf.cast(a,'float16'))
        inputList_dark_trainData = tf.data.Dataset.from_tensor_slices(inputList_dark_trainData,'inputList_dark_trainData').map(lambda a : tf.cast(a,'float16'))
        inputList_normal_trainData = tf.data.Dataset.from_tensor_slices(inputList_normal_trainData,'inputLIst_normal_trainData').map(lambda a : tf.cast(a,'float16'))

        inputList_light_testData = tf.data.Dataset.from_tensor_slices(inputList_light_testData,'inputList_light_testData').map(lambda a : tf.cast(a,'float16'))
        inputList_dark_testData = tf.data.Dataset.from_tensor_slices(inputList_dark_testData,'inputList_dark_testData').map(lambda a : tf.cast(a,'float16'))
        inputList_normal_testData = tf.data.Dataset.from_tensor_slices(inputList_normal_testData,'inputLIst_normal_testData').map(lambda a : tf.cast(a,'float16'))
        '''
        return [inputList_light_trainData[:allNumber],inputList_dark_trainData[:allNumber],inputList_normal_trainData[:allNumber],
                inputList_light_testData[:allNumber],inputList_dark_testData[:allNumber],inputList_normal_testData[:allNumber]]
    


    def modelSave(self,number,suffix='keras'):
        self.model.save(f'{modelSave_path}model{number}-GnonEn.{suffix}')
        self.discriminator.save(f'{modelSave_path}model{number}-D.{suffix}')
        return


class lossFunction_generator(tf.keras.losses.Loss):
    def call(self,trueImage,outImage,weight):
        MSl = tf.reduce_mean(tf.keras.metrics.mean_squared_error(trueImage,outImage))
        ssim = tf.reduce_sum(tf.image.ssim(trueImage,outImage,255))
        loss = weight[0] * MSl + weight[1] * ssim
        return [loss,MSl,ssim]



class lossFunction_meanSquared(tf.keras.losses.Loss):
    def call(self,trueImage,outImage):
        Y = tf.reduce_mean(tf.keras.metrics.mean_squared_error(trueImage,outImage))
        y = len(trueImage)
        #Y = tf.reduce_sum((trueImage - outImage) ** 2) / y
        return Y



class lossFunction_ssim(tf.keras.losses.Loss):
    def call(self,trueImage,outImage):
        return tf.reduce_sum(tf.image.ssim(trueImage,outImage,255))



class lossFunction_crossEntropy(tf.keras.losses.Loss):                          #交叉熵损失函数
    def call(self,outImage,trueImage):
        #Y = -tf.reduce_sum(trueImage_discriminatorOutput * tf.math.log(tf.math.softmax(outImage_discriminatorOutput)),-1)
        #Y = -(tf.reduce_sum(trueImage * tf.math.log(outImage) + (1 - trueImage) * tf.math.log(1 - outImage)) / tf.cast(tf.reduce_sum(trueImage.shape),'float32'))
        Y1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(trueImage,outImage))
        return Y1


'''
#与wgan-gp有关的代码，应该用不到了
class lossFunction_gradientPenalty(tf.keras.losses.Loss):       #wgangp里的梯度惩罚
    def call(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        # 求y_pred关于averaged_samples的导数（梯度）
        # 即判别器的判断结果validity_interpolated与加权样本interpolated_img求导
        # ps：interpolated_img作为默认参数（averaged_samples）在使用partial封装时已经提供
        gradients = k.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...,计算范数
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        # 基本上就是对论文中的公式的实现
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return tf.reduce_mean(gradient_penalty)
    


class lossFunction_wasserstein(tf.keras.losses.Loss):           #计算em距离，用于生成器
    def call(self,trueImage,outImage):
        ''
        沿着指定轴取张量的平均值，
        得到一个具有y_true * y_pred元素均值的张量
        ''
        return tf.reduce_mean(trueImage * outImage)
    


# 定义WGAN-GP的损失函数
LAMBDA = 10  # 梯度惩罚的权重

@tf.function
def WGAN_GP_train_d_step(real_image, batch_size, step):
    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator([noise], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator([fake_image_mixed], training=True)
            
            # 计算梯度惩罚
            grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
            grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
            
            fake_pred = discriminator([fake_image], training=True)
            real_pred = discriminator([real_image], training=True)
            
            D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty

        # 计算判别器的梯度
        D_gradients = d_tape.gradient(D_loss, discriminator.trainable_variables)
        D_optimizer.apply_gradients(zip(D_gradients, discriminator.trainable_variables))

@tf.function
def WGAN_GP_train_g_step(real_image, batch_size, step):
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as g_tape:
        fake_image = generator([noise], training=True)
        fake_pred = discriminator([fake_image], training=True)
        G_loss = -tf.reduce_mean(fake_pred)

        # 计算生成器的梯度
        G_gradients = g_tape.gradient(G_loss, generator.trainable_variables)
        G_optimizer.apply_gradients(zip(G_gradients, generator.trainable_variables))


全部自己完成的每步训练函数，已废弃
    def train_step(self,inputList,trueImage):         #定义训练每步内容
        #先训练鉴别器
        genertedImage = self.model(inputList)                   #调用生成器
        print(genertedImage[0][0][:3])

        with tf.GradientTape() as discriminatorTape:         #自动进行梯度计算，之后用于更新参数
            
            realOutput = self.discriminator(trueImage)              #调用鉴别器
            falseOutput = self.discriminator(genertedImage)
            print(realOutput,falseOutput)

            discriminatorLoss1 = lossFunction_crossEntropy().call(realOutput,tf.ones_like(realOutput,'float32'))        #鉴别器的两个损失
            discriminatorLoss2 = lossFunction_crossEntropy().call(falseOutput,tf.zeros_like(falseOutput,'float32'))
            discriminatorLoss = (discriminatorLoss1 + discriminatorLoss2) / 2

        discriminatorGradients = discriminatorTape.gradient(discriminatorLoss,self.discriminator.trainable_variables)    
        self.discriminatorOptimizer.apply_gradients(zip(discriminatorGradients,self.discriminator.trainable_variables))
        self.discriminator.trainable = False
        with tf.GradientTape(persistent=True) as generatorTape:
            #后训练生成器
            genertedImage = self.model(inputList)
            falseOutput_update = self.discriminator(genertedImage)
            generatorLoss_MS = lossFunction_meanSquared().call(trueImage,genertedImage)         #生成器损失函数
            generatorLoss_CE = lossFunction_crossEntropy().call(falseOutput_update,tf.ones_like(realOutput,'float32'))
            generatorLoss_ssim = lossFunction_ssim().call(trueImage,genertedImage)
            generatorLoss = generatorLoss_MS * 0.1 + generatorLoss_ssim * 10 + generatorLoss_CE * 5
        
        generatorGradients = generatorTape.gradient(generatorLoss,self.model.trainable_variables)                   #获取梯度        
        self.generatorOptimizer.apply_gradients(zip(generatorGradients,self.model.trainable_variables))             #更新参数
        self.discriminator.trainable = True
        #print(f'Gloss: {generatorLoss} ,(MS:{generatorLoss_MS},ssim:{generatorLoss_ssim}), Dloss: {discriminatorLoss} ')
        print(f'Gloss: {generatorLoss} Dloss: {discriminatorLoss} ')
        return 
'''