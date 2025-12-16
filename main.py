import inputImage
import ResCNN
import visionTransformer
import EnhanceAndDecoder
import train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


'''Main関数'''


imgPath = './test_imgs/'        #入力画像の保存パス

def test():                    #テスト用関数
    lightList = inputImage.openFunction(imgPath + 'o1.JPG')     #分割後の画像結果を返す
    darkList = inputImage.openFunction(imgPath + 'u1.JPG')

    lightRes = ResCNN.openFunction(lightList)                   #残差計算を行う
    darkRes = ResCNN.openFunction(darkList)

    ViT = visionTransformer.openFunction(lightList,darkList)                #ViTの計算を行う
    generateImg = EnhanceAndDecoder.openFunction(lightRes,darkRes,ViT)      画像を生成する生成图像

    discriminatorLight = visionTransformer.openFunction_discriminator(lightList,darkList,generateImg)    #識別器に入力して判定を行う


if __name__ == "__main__":
    train.training().call()
    #evaluation.test().call()

