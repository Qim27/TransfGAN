import tkinter.font
from PIL import Image,ImageTk
import tensorflow as tf
import numpy
import tkinter
import tkinter.filedialog
import imageio

import inputImage

#UI画面

top = tkinter.Tk()
top.geometry('1200x650')
top.title('多曝光融合系统可视化界面')
top.grid_rowconfigure(0,weight=3)
top.grid_columnconfigure(0,weight=0)
top.grid_rowconfigure(1,weight=10)
top.grid_columnconfigure(1,weight=10)
top.grid_rowconfigure(2,weight=0)
top.grid_columnconfigure(2,weight=0)
top.grid_rowconfigure(3,weight=10)
top.grid_columnconfigure(3,weight=5)
top.grid_rowconfigure(4,weight=3)
top.grid_columnconfigure(4,weight=10)

img = None
img2 = None
modelPath = None
outImage = None
imgPath = [None,None]
inputImage_list = [None,None]

def pathInput1():
    global img,imgPath
    imgPath[0] = tkinter.filedialog.askopenfilename()
    imgInp = Image.open(imgPath[0])
    img = ImageTk.PhotoImage(imgInp)
    labelImg = tkinter.Label(top,text='img',fg='#66ccff',image=img)
    labelImg.grid(row=1,column=1)
    

def pathInput2():
    global img2
    imgPath[1] = tkinter.filedialog.askopenfilename()
    imgInp = Image.open(imgPath[1])
    img2 = ImageTk.PhotoImage(imgInp)
    #print(imgPath,img2)
    labelImg = tkinter.Label(top,text='img',fg='#66ccff',image=img2)
    labelImg.grid(row=3,column=1)


def pathInput3():
    global modelPath
    modelPath = tkinter.filedialog.askdirectory()
    modelPath_list = modelPath.split('/')
    labelText = tkinter.Label(top,text=modelPath_list[-1],font=fontSize2)
    labelText.grid(row=2,column=3)


def runModel():
    global outImage,imgPath,modelPath
    model = tf.saved_model.load(modelPath)
    inputImg_light = numpy.array(imageio.v3.imread(imgPath[1]),'float32')
    inputImg_dark = numpy.array(imageio.v3.imread(imgPath[0]),'float32')
    inputImg_light = inputImg_light[numpy.newaxis,:,:,:]
    inputImg_dark = inputImg_dark[numpy.newaxis,:,:,:]
    #numpy.array(inputList_dark[trainSize : ],'float32')
    print(inputImg_dark.shape,inputImg_light.shape)
    
    outImage = model([inputImg_light,inputImg_dark])
    inputImage.saveImage_tensorImg(outImage,'./temp')
    outImage = ImageTk.PhotoImage(Image.open('./tempimg0.jpg'))
    labelImg = tkinter.Label(top,text='img',image=outImage)
    labelImg.grid(row=1,column=4)
    
# empty1 = tkinter.Label(top,text='    ')
# empty1.grid(row=0,column=0)
# empty2 = tkinter.Label(top,text='    ')
# empty2.grid(row=0,column=2)

fontSize = tkinter.font.Font(family='等线',size=30)
fontSize2 = tkinter.font.Font(family='等线',size=20)

label1 = tkinter.Label(top,text='多曝光融合系统可视化界面',font=fontSize)
label1.grid(row=0,column=3)
inp1 = tkinter.Button(top,text='选择欠曝图',command=pathInput1,font=fontSize2,fg='#11659a',relief=tkinter.SOLID)
inp1.grid(row=2,column=1)
inp2 = tkinter.Button(top,text='选择过曝图',command=pathInput2,font=fontSize2,fg='#11659a',relief=tkinter.SOLID)
inp2.grid(row=4,column=1)
inp3 = tkinter.Button(top,text='选择模型',command=pathInput3,font=fontSize2,fg='#11659a',relief=tkinter.SOLID)
inp3.grid(row=3,column=3)
inp4 = tkinter.Button(top,text='开始融合',command=runModel,font=fontSize2,fg='#11659a',relief=tkinter.SOLID)
inp4.grid(row=2,column=4)


#labelImg.pack()


top.mainloop()
