import inputImage
import ResCNN
import visionTransformer
import EnhanceAndDecoder
import train
#import evaluation
#import comparisonCode.传统算法2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



'''main函数'''


imgPath = './test_imgs/'        #输入照片存储路径

def test():
    lightList = inputImage.openFunction(imgPath + 'o1.JPG')     #返回分割后的图片结果
    darkList = inputImage.openFunction(imgPath + 'u1.JPG')

    lightRes = ResCNN.openFunction(lightList)                   #进行残差计算
    darkRes = ResCNN.openFunction(darkList)

    ViT = visionTransformer.openFunction(lightList,darkList)                #进行ViT计算
    generateImg = EnhanceAndDecoder.openFunction(lightRes,darkRes,ViT)      #生成图像

    discriminatorLight = visionTransformer.openFunction_discriminator(lightList,darkList,generateImg)    #输入进鉴别器进行判定



if __name__ == "__main__":
    train.training().call()
    #evaluation.test().call()
    #comparisonCode.传统算法2.mmmm()
    #test()