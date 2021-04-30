import wx
import os
import cv2
import dlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import math
import sys
import pylab
from scipy import signal
import scipy as sp
import io
from sklearn import datasets
import csv
import skimage
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

class MyFrame(wx.Frame):
    #initialize a frame and add components on it
    def __init__(self):
        wx.Frame.__init__(self,parent=None,id=wx.ID_ANY,title="Glasses Recognition",size=(1000,700))
        self.InitUI()
        self.Center()
        self.Show()

    def InitUI(self):
        self.panel=wx.Panel(self)
        self.sizer = wx.GridBagSizer(10,20)
        btn1 = wx.Button(self.panel,-1,"Open")
        btn2 = wx.Button(self.panel,-1,"Recognize")
        text1 = wx.StaticText(self.panel, label="Original Image")
        text2 = wx.StaticText(self.panel, label="Result")
        self.text3 = wx.StaticText(self.panel, label="with glasses? :  ")
        #self.text5 = wx.StaticText(self.panel, label="training accuracy :  ")
        self.text4 = wx.StaticText(self.panel, label=" ")
        #self.text6 = wx.StaticText(self.panel, label=" ")
        self.sizer.Add(text1,(2,1),(1,1),flag=wx.ALL,border=10)
        self.sizer.Add(text2,(2,4),(1,1),flag=wx.ALL,border=10)
        self.sizer.Add(btn1, (3,8),(1,1), border=5)
        self.sizer.Add(btn2, (4,8),(1,1), border=5)
        self.sizer.Add(self.text3,(5,8),(1,1),border=5)
        self.sizer.Add(self.text4,(5,9),(1,1),border=5)
        #self.sizer.Add(self.text5,(5,8),(1,1),border=5)
        #self.sizer.Add(self.text6,(5,9),(1,1),border=5)
        btn1.Bind(wx.EVT_BUTTON, self.Open)
        btn2.Bind(wx.EVT_BUTTON, self.Recognize)
        self.panel.SetSizerAndFit(self.sizer)

    #align the face in the image to horizontal level
    def face_alignment(self,faces):
        #initialize the face detection library and two lists
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
        faces_aligned = []
        edclideans = []
        #operate the faces in the image one by one
        for face in faces:
            rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
            shape = predictor(np.uint8(face),rec)
            #calculate the location of eye center for rotation
            order = [36,39,42,45,30,48,54] 
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

            eye_center =((shape.part(36).x + shape.part(45).x) * 1./2,
                        (shape.part(36).y + shape.part(45).y) * 1./2)
            dx = (shape.part(45).x - shape.part(36).x) 
            dy = (shape.part(45).y - shape.part(36).y)
            euclideandis = math.sqrt(dx*dx + dy*dy)
            edclideans.append(euclideandis)

            angle = math.atan2(dy,dx) * 180. / math.pi 
            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) 
            #rotate the face according to the angle 
            RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) 
            #collect all the faces
            faces_aligned.append(RotImg)
        return faces_aligned


    def get_pixel(self,img, center, x, y):
        new_value = 0
        try:
            #compare neighbor pixels with center pixel
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    #changed LBP algorism, the center pixel is added a weight
    def lbp_calculated_pixel(self,img, x, y):
        center = img[x][y]
        w, h = img.shape
        WLG = 0
        if x==0 and y !=0 and y != h - 1:
            WLG = ((img[x][y+1] + img[x+1][y+1] + img[x+1][y] + img[x+1][y-1] + 
            img[x][y-1]) + 25*center)/(8+25)
        elif x == w - 1 and y !=0 and y != h - 1:
            WLG = ((img[x-1][y+1] + img[x][y+1] + 
            img[x][y-1] + img[x-1][y-1] + img[x-1][y]) + 25*center)/(8+25)
        elif y==0 and x !=0 and x != w - 1:
            WLG = ((img[x-1][y+1] + img[x][y+1] + img[x+1][y+1] + img[x+1][y] + 
            img[x-1][y]) + 25*center)/(8+25)
        elif y==h -1 and x !=0 and x != w -1:
            WLG = ((img[x+1][y] + img[x+1][y-1] + 
            img[x][y-1] + img[x-1][y-1] + img[x-1][y]) + 25*center)/(8+25)
        elif x == 0  and y == 0:
            WLG = (img[x][y+1] + img[x+1][y+1] + img[x+1][y] + 
            + 25*center)/(8+25)
        elif x == 0 and y == h - 1:
            WLG = ((img[x+1][y] + img[x+1][y-1] + 
            img[x][y-1]) + 25*center)/(8+25)
        elif x == w -1  and y == 0:
            WLG = ((img[x-1][y+1] + img[x][y+1] + 
            img[x-1][y]) + 25*center)/(8+25)
        elif x == w -1 and y == h -1:
            WLG = ((img[x][y-1] + img[x-1][y-1] + img[x-1][y]) + 25*center)/(8+25)    
        else:
            WLG = ((img[x-1][y+1] + img[x][y+1] + img[x+1][y+1] + img[x+1][y] + img[x+1][y-1] + 
            img[x][y-1] + img[x-1][y-1] + img[x-1][y]) + 25*center)/(8+25)
        #collect the eight pixel values around center pixel
        val_ar = []
        val_ar.append(self.get_pixel(img, WLG, x-1, y+1))     # top_right
        val_ar.append(self.get_pixel(img, WLG, x, y+1))       # right
        val_ar.append(self.get_pixel(img, WLG, x+1, y+1))     # bottom_right
        val_ar.append(self.get_pixel(img, WLG, x+1, y))       # bottom
        val_ar.append(self.get_pixel(img, WLG, x+1, y-1))     # bottom_left
        val_ar.append(self.get_pixel(img, WLG, x, y-1))       # left
        val_ar.append(self.get_pixel(img, WLG, x-1, y-1))     # top_left
        val_ar.append(self.get_pixel(img, WLG, x-1, y))       # top
        #transfer the data to 0~255
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val

    #get the corner location of the target area
    def rect_to_bb(self,rect): 
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    #read the chose image and transfer them to LBP images
    def readImg(self, image):
        im_raw = cv2.resize(image,(300,300),interpolation=cv2.INTER_CUBIC)
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        src_faces = []
        #align the faces according to the eye corners location
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = self.rect_to_bb(rect)
            detect_face = gray[y:y+h,x:x+w]          
            src_faces.append(detect_face)
        faces_aligned = self.face_alignment(src_faces)
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
        rec = dlib.rectangle(0,0,faces_aligned[0].shape[0],faces_aligned[0].shape[1])
        shape = predictor(np.uint8(faces_aligned[0]),rec) 
        order = [36,39,42,45] 
        xe = int((shape.part(36).x + shape.part(45).x) * 1./2)
        ye = int((shape.part(36).y + shape.part(45).y) * 1./2)
        #get the region of interest
        xl = 0
        yt = 12
        xr = 186
        yb = 100
        roi = faces_aligned[0][yt:yb,xl:xr]
        #cv2.imshow(roi)
        #get the size of roi and map the value of the pixel to lbp value
        height, width = roi.shape
        img_lbp = np.zeros((height, width,3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = self.lbp_calculated_pixel(roi, i, j)
        return img_lbp

    #read the image information in the folder and generate a collection list
    def generate_dataset(self,path):
        filelist = os.listdir(path)
        csvfile = open("imgfile_trainprocess.txt", 'w')
        for files in filelist:
            filename = os.path.splitext(files)[0]
            str1 = path + files + ' ' + filename[0] + '\n'
            csvfile.writelines(str1)
        csvfile.close()
        return csvfile.name

    #generate the training datasets for sklearn
    def load_imgesets(self,filename):
        file = open(filename,'r')
        data = []
        target = []
        data = np.array(data,dtype=float)
        flag = 1
        for line in file:
            #split the path and class of a image
            str = line.split(' ',1)
            #transfer the grayscale image into one dimension array
            if flag == 1:
                flag = 0
                data_i = np.array(Image.open(str[0]).convert('L')).reshape(1,-1)
                pixcot = np.zeros((256,),np.int32)
                #calculate the histogram for the image
                for index, value in enumerate(data_i):
                    pixcot[value] += 1
                data = pixcot
            else:
                row = np.array(Image.open(str[0]).convert('L')).reshape(1, -1)
                pixcot = np.zeros((256,),np.int32)
                for index, value in enumerate(data_i):
                    pixcot[value] += 1
                row = pixcot
                data = np.row_stack((data, row))
            #collect all the classes into one list
            target.append(str[1])
        file.close()
        target =np.asarray(target,dtype=int)
        return data,target

    #generate dataset for sklearn
    def image_datasets(self,path):
        filename = self.generate_dataset(path)
        data,target = self.load_imgesets(filename)
        return data,target

    #convert BGR type to RGB type
    def BGR2RGB(self,src):
        (B,G,R) = cv2.split(src)
        img=cv2.merge([R,G,B])
        return img

    #open an image from dialog box and show it on GUI
    def Open(self, event):
        msg=wx.FileDialog(self,message="Open an image",defaultDir=os.getcwd(),style=wx.FD_OPEN)
        ret = msg.ShowModal()
        if ret == wx.ID_OK:
            p = msg.GetPath()
            self.orig = wx.Image(p,wx.BITMAP_TYPE_ANY).Scale(300,300).ConvertToBitmap()
            orig2 = wx.Image(p,wx.BITMAP_TYPE_ANY).Scale(186,88).ConvertToBitmap()
            self.arr = cv2.imread(p).astype('uint8')     
            img1 = wx.StaticBitmap(self.panel, -1, wx.Bitmap(self.orig))
            self.img2 = wx.StaticBitmap(self.panel, -1, wx.Bitmap(orig2))
            self.sizer.Add(img1, (3,1), (3,3),flag=wx.ALL, border=0)
            self.sizer.Add(self.img2, (3,4), (3,3),flag=wx.ALL, border=0)
            self.panel.SetSizerAndFit(self.sizer)
                #generate data and target according to the path of training images
        target_directory = 'C:/Users/Jerry/VS/project/trainingprocess/isglass/'
        d, t = self.image_datasets(target_directory)
        indices = np.random.permutation(len(d))
        #use 80% of the images for training and the left for test
        d_train = d[indices[:-20]]
        t_train = t[indices[:-20]]
        d_test = d[indices[-20:]]
        t_test = t[indices[-20:]]
        #define a svm as classifier
        self.clf = svm.SVC(kernel='rbf',gamma='auto',decision_function_shape='ovr')
        self.clf.fit(d,t)
        #show the training accuracy
        #predict = clf.predict(d_test)
        #print(predict)
        score = self.clf.score(d_test,t_test)
        #self.text6.SetLabel(str(score))
        print('Training Accuracy: ', score)
    
    #recognize the selected image
    def Recognize(self,event):
        lbp = self.readImg(self.arr)
        cv2.imwrite('C:/Users/Jerry/VS/project/thereadimages/' +'a.jpg', lbp)
        display = wx.Image('C:/Users/Jerry/VS/project/thereadimages/a.jpg',wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.img2.SetBitmap(display)

        data0 = []
        data1 = np.array(data0,dtype=float)
        #transfer the grayscale image into one dimension array
        data_i = np.array(Image.open('C:/Users/Jerry/VS/project/thereadimages/a.jpg').convert('L')).reshape(1,-1)
        pixcot = np.zeros((256,),np.int32)
        #calculate the histogram for the image
        for index, value in enumerate(data_i):
            pixcot[value] += 1
        data1 = np.array(pixcot).reshape(1,-1)
        #show the predict result
        result = self.clf.predict(data1)
        if result == 1:
            self.text4.SetLabel('with glasses')
        elif result == 2:
            self.text4.SetLabel('no glasses')

if __name__=='__main__':
    app=wx.App()
    MyFrame()
    app.MainLoop()