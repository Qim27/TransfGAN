import imageio
import os
import numpy as np


'''读取/保存图片函数/画像の読み取りと保存関数'''

#调用函数/API
def openFunction(imgPath):
    return inImage(imgPath)


#读取/裁剪图片/画像の読み取りとカット
def inImage(imgPath,imgWeith=256,imgLength=256,wholeImage=False):
    imageTemp1 = imageio.v3.imread(imgPath)     #读取图片
    height,width,_ = imageTemp1.shape           #获取图像各项信息
    weimax = int(width / imgWeith)              #计算最大可分的数量
    heimax = int(height / imgLength)

    if(wholeImage):
        return imageTemp1

    i,j,imglist = 0,0,[]
    for i in range(0,heimax):        #循环分割为256 * 256大小的图像
        for j in range(0,weimax):
            img = imageTemp1[i * imgWeith : (i + 1) * imgWeith,j * imgLength : (j + 1) * imgLength , : ]
            imglist.append(img)   
    return imglist



def inImage_simple(imgPath='./outImages/',amount=15):
    inputList = []
    for i in range(amount):
        inputList.append(imageio.v3.imread(f'{imgPath}img{i}.jpg'))
    return inputList


def inImage_simple_temp(imgPath='./outImages/',amount=15):
    inputList = []
    for i in range(amount):
        inputList.append(imageio.v3.imread(f'{imgPath}mrf_img{i}.jpg'))
    return inputList



#保存图片/画像の保存
def saveImage(imgArray,savePath='./outImages/'):
    if not os.path.exists(savePath):       
        os.makedirs(savePath)
        
    batchSize = len(imgArray)
    for i in range(batchSize):
        Y = imgArray[i].astype(np.uint8)
        imageio.v3.imwrite(f'{savePath}img{i}.jpg',Y)


def saveImage_tensorImg(tensorImg,savePath='./outImages/'):
    if not os.path.exists(savePath):       
        os.makedirs(savePath)

    batchSize,_,_,_ = tensorImg.shape
    for i in range(batchSize):
        Y = tensorImg[i]                #改变类型和数据类型
        Y = np.around(Y.numpy(),3).astype(np.uint8)
        imageio.v3.imwrite(f'{savePath}img{i}.jpg',Y,extension='.jpg')



