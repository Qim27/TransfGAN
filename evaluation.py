import skimage
import sklearn.metrics 
import numpy as np
import math
import cv2
import tensorflow as tf

import inputImage
import train



'''评价对比用'''



allNumber = 15
testImang_filerPath = './Dataset_Part1/'
modelSave_path = './model/'

class test():                   #评估模型效果用/再输出用
    def call(self,EVAMODE=True):
        if(EVAMODE):
            inputNormal = np.array(inputImage.inImage_simple('./outImages/normal/',20),'float32')
            outImage = np.array(inputImage.inImage_simple('./outImages/',20),'float32')
            #outImage = np.array(inputImage.inImage_simple_temp('./comparisonCode/comparisonImage2/',20),'float32')
            #inputNormal = np.array(inputImage.inImage_simple_temp('./Dataset_Part1/1/44.JPG',1),'float32')
            #outImage = np.array(inputImage.inImage_simple_temp('./comparisonCode/0_0_0.png',1),'float32')
            amount = len(outImage)
            ssimLossA,psnrLossA,mseLossA,entropyLossA,AGLossA,STDLossA1,STDLossA2,MILossA = [0,0,0,0,0,0,0,0]

            for i in range(amount):             #计算不同评估指标
                inputNormal_one = inputNormal[i]
                outImage_one = outImage[i]
                ssimLoss = evaluationIndicator().ssim(inputNormal_one,outImage_one)
                psnrLoss = evaluationIndicator().psnr(inputNormal_one,outImage_one)
                mseLoss = evaluationIndicator().mse(inputNormal_one,outImage_one)
                entropyLoss = evaluationIndicator().entropy(outImage_one)
                AGLoss = evaluationIndicator().AG(outImage_one)
                STDLoss = evaluationIndicator().STD(outImage_one)
                MILoss = evaluationIndicator().MI(inputNormal_one,outImage_one)
                ssimLossA += ssimLoss
                psnrLossA += psnrLoss
                mseLossA += mseLoss
                entropyLossA += entropyLoss
                AGLossA += AGLoss
                STDLossA1 += STDLoss[0]
                STDLossA2 += STDLoss[1]
                MILossA += MILoss

                print(f'{i+1}/{amount}. 各指标情况')
                print(f'ssim:{ssimLoss:.4f}, 峰值信噪比:{psnrLoss:.3f}, 均方误差:{mseLoss:.2f}, 信息熵:{entropyLoss:.3f}, 平均梯度:{AGLoss:.2f}, 均值:{STDLoss[0]:.2f}, 标准差:{STDLoss[1]:.2f}, 互信息:{MILoss:.4f}')

            ssimLossA /= amount
            psnrLossA /= amount
            mseLossA /= amount
            entropyLossA /= amount
            AGLossA /= amount
            STDLossA1 /= amount
            STDLossA2 /= amount
            MILossA /= amount
            print(f'总计: ssim:{ssimLossA:.4f}, 峰值信噪比:{psnrLossA:.3f}, 均方误差:{mseLossA:.2f}, 信息熵:{entropyLossA:.3f}, 平均梯度:{AGLossA:.2f}, 均值:{STDLossA1:.2f}, 标准差:{STDLossA2:.2f}, 互信息:{MILossA:.4f}')
            print('evaluation end')
            return

        inputList_light_trainData,inputList_dark_trainData,inputList_normal_trainData,_,_,_ = train.training().inputImage_all(testImang_filerPath)
        
        model = tf.saved_model.load(f'{modelSave_path}model19-GOnly.tf')
        outImage = model([inputList_light_trainData,inputList_dark_trainData])
        inputImage.saveImage_tensorImg(outImage,'./evaluationImgs/outImgs/')
        inputImage.saveImage(inputList_light_trainData,'./evaluationImgs/lightImgs/')
        inputImage.saveImage(inputList_dark_trainData,'./evaluationImgs/darkImgs/')
        inputImage.saveImage(inputList_normal_trainData,'./evaluationImgs/normalImgs/')
        print('creating end')



class evaluationIndicator():
    def ssim(self,trueImage,outImage):      #结构相似性
        #Y = tf.keras.metrics.mean_squared_error(trueImage,outImage)
        Y = skimage.metrics.structural_similarity(trueImage,outImage,data_range=255.0,channel_axis=2)
        return tf.reduce_mean(Y)
    

    def psnr(self,trueImage,outImage):      #峰值信噪比
        return skimage.metrics.peak_signal_noise_ratio(trueImage,outImage,data_range=255)
    

    def mse(self,trueImage,outImage):       #均方误差
        return skimage.metrics.mean_squared_error(trueImage,outImage)
    

    def entropy(self,outImage):             #信息熵
        return skimage.measure.shannon_entropy(outImage,2)
    

    def AG(self,outImage):                  #平均梯度
        height,width,c = outImage.shape
        tmp = 0.0

        for i in range(height-1):
            for j in range(width-1):
                dx = outImage[i,j+1]-outImage[i,j]
                dy = outImage[i+1,j]-outImage[i,j]
                ds = tf.sqrt((dx*dx+dy*dy)/2)
                tmp += ds
        
        imageAG = tmp/(width*height)
        return tf.reduce_sum(imageAG /c)
    

    def STD(self,outImage):                 #均值和标准差
        mean,stddv = cv2.meanStdDev(outImage)
        return [tf.reduce_sum(mean),tf.reduce_sum(stddv)]
    

    def MI(self,trueImage,outImage):        #互信息
        trueImage = np.reshape(trueImage,-1)
        outImage = np.reshape(outImage,-1)
        MI = sklearn.metrics.mutual_info_score(trueImage,outImage)
        return MI
    

    def Qabf():
        #model parameters 模型参数
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8

        #Sobel Operator Sobel算子
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        #if y is the response to h1 and x is the response to h3;then the intensity is sqrt(x^2+y^2) and  is arctan(y/x);
        #如果y对应h1，x对应h2，则强度为sqrt(x^2+y^2)，方向为arctan(y/x)

        strA = cv2.imread("images/MDDataset/c01_1.tif", 0).astype(np.float32)
        strB = cv2.imread("images/MDDataset/c01_2.tif", 0).astype(np.float32)
        strF = cv2.imread("results/our/guided_1.png", 0).astype(np.float32)


        # 数组旋转180度
        def flip180(arr):
            new_arr = arr.reshape(arr.size)
            new_arr = new_arr[::-1]
            new_arr = new_arr.reshape(arr.shape)
            return new_arr

        #相当于matlab的Conv2
        def convolution(k, data):
            k = flip180(k)
            data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
            n,m = data.shape
            img_new = []
            for i in range(n-2):
                line = []
                for j in range(m-2):
                    a = data[i:i+3,j:j+3]
                    line.append(np.sum(np.multiply(k, a)))
                img_new.append(line)
            return np.array(img_new)


        #用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
        #matlab会对图像进行补0，然后卷积核选择180度
        #gA = sqrt(SAx.^2 + SAy.^2);
        #定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
        def getArray(img):
            SAx = convolution(h3,img)
            SAy = convolution(h1,img)
            gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
            n, m = img.shape
            aA = np.zeros((n,m))
            for i in range(n):
                for j in range(m):
                    if(SAx[i,j]==0):
                        aA[i,j] = math.pi/2
                    else:
                        aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
            return gA,aA

        #对strB和strF进行相同的操作
        gA,aA = getArray(strA)
        gB,aB = getArray(strB)
        gF,aF = getArray(strF)

        #the relative strength and orientation value of GAF,GBF and AAF,ABF;
        def getQabf(aA,gA,aF,gF):
            n, m = aA.shape
            GAF = np.zeros((n,m))
            AAF = np.zeros((n,m))
            QgAF = np.zeros((n,m))
            QaAF = np.zeros((n,m))
            QAF = np.zeros((n,m))
            for i in range(n):
                for j in range(m):
                    if(gA[i,j]>gF[i,j]):
                        GAF[i,j] = gF[i,j]/gA[i,j]
                    elif(gA[i,j]==gF[i,j]):
                        GAF[i, j] = gF[i, j]
                    else:
                        GAF[i, j] = gA[i,j]/gF[i, j]
                    AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

                    QgAF[i,j] = Tg/(1+math.exp(kg*(GAF[i,j]-Dg)))
                    QaAF[i,j] = Ta/(1+math.exp(ka*(AAF[i,j]-Da)))

                    QAF[i,j] = QgAF[i,j]*QaAF[i,j]

            return QAF

        QAF = getQabf(aA,gA,aF,gF)
        QBF = getQabf(aB,gB,aF,gF)


        #计算QABF
        deno = np.sum(gA+gB)
        nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))
        output = nume/deno
        print(output)