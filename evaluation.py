import skimage
import sklearn.metrics 
import numpy as np
import math
import cv2
import tensorflow as tf

import inputImage
import train



'''モデル評価・比較用'''



allNumber = 15
testImang_filerPath = './Dataset_Part1/'
modelSave_path = './model/'

class test():                   #モデル性能評価／結果出力用
    def call(self,EVAMODE=True):
        if(EVAMODE):
            inputNormal = np.array(inputImage.inImage_simple('./outImages/normal/',20),'float32')
            outImage = np.array(inputImage.inImage_simple('./outImages/',20),'float32')
            #outImage = np.array(inputImage.inImage_simple_temp('./comparisonCode/comparisonImage2/',20),'float32')
            #inputNormal = np.array(inputImage.inImage_simple_temp('./Dataset_Part1/1/44.JPG',1),'float32')
            #outImage = np.array(inputImage.inImage_simple_temp('./comparisonCode/0_0_0.png',1),'float32')
            amount = len(outImage)
            ssimLossA,psnrLossA,mseLossA,entropyLossA,AGLossA,STDLossA1,STDLossA2,MILossA = [0,0,0,0,0,0,0,0]

            for i in range(amount):             #各種評価指標を計算
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

                print(f'{i+1}/{amount}. 各指標の結果')
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


#各種損失関数
class evaluationIndicator():
    def ssim(self,trueImage,outImage):      #構造類似度（SSIM）
        #Y = tf.keras.metrics.mean_squared_error(trueImage,outImage)
        Y = skimage.metrics.structural_similarity(trueImage,outImage,data_range=255.0,channel_axis=2)
        return tf.reduce_mean(Y)
    

    def psnr(self,trueImage,outImage):      #ピーク信号対雑音比（PSNR）
        return skimage.metrics.peak_signal_noise_ratio(trueImage,outImage,data_range=255)
    

    def mse(self,trueImage,outImage):       #平均二乗誤差（MSE）
        return skimage.metrics.mean_squared_error(trueImage,outImage)
    

    def entropy(self,outImage):             #情報エントロピー
        return skimage.measure.shannon_entropy(outImage,2)
    

    def AG(self,outImage):                  #平均勾配
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
    

    def STD(self,outImage):                 #平均値と標準偏差
        mean,stddv = cv2.meanStdDev(outImage)
        return [tf.reduce_sum(mean),tf.reduce_sum(stddv)]
    

    def MI(self,trueImage,outImage):        #相互情報量
        trueImage = np.reshape(trueImage,-1)
        outImage = np.reshape(outImage,-1)
        MI = sklearn.metrics.mutual_info_score(trueImage,outImage)
        return MI
