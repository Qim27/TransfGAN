import imageio
import os
import numpy as np


'''画像の読み込み／保存関数'''

#API
def openFunction(imgPath):
    return inImage(imgPath)


#画像の読み込み／切り出し
def inImage(imgPath,imgWeith=256,imgLength=256,wholeImage=False):
    imageTemp1 = imageio.v3.imread(imgPath)     #画像を読み込む
    height,width,_ = imageTemp1.shape           #画像の各種情報を取得
    weimax = int(width / imgWeith)              #分割可能な最大数を計算
    heimax = int(height / imgLength)

    if(wholeImage):
        return imageTemp1

    i,j,imglist = 0,0,[]
    for i in range(0,heimax):        #256×256 サイズの画像に分割するループ
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



#画像の保存
def saveImage(imgArray,savePath='./outImages/'):
    if not os.path.exists(savePath):       #保存パスが存在しない場合は作成
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
        Y = tensorImg[i]                #型およびデータ形式を変更
        Y = np.around(Y.numpy(),3).astype(np.uint8)
        imageio.v3.imwrite(f'{savePath}img{i}.jpg',Y,extension='.jpg')




