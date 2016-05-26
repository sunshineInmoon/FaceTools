# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:13:58 2015

@author: Administrator
"""

import numpy as np
import cv2
import os

'''
函数：FaceDetct()
函数功能：人脸检测
输入参数：in_name----输入图片路径
         out_name----输出图片路径
'''
def FaceDetect(in_name,out_name,new_w,new_h):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(in_name)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detects objects of different sizes in the input image.
# The detected objects are returned as a list of rectangles.
#cv2.CascadeClassifier.detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
#scaleFactor – Parameter specifying how much the image size is reduced at each image
#scale.
#minNeighbors – Parameter specifying how many neighbors each candidate rectangle should
#have to retain it.
#minSize – Minimum possible object size. Objects smaller than that are ignored.
#maxSize – Maximum possible object size. Objects larger than that are ignored.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = img[y:y+h,x:x+w]
        #重新变换大小后在存储
        res = cv2.resize(roi,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_name,res)
        #显示检测结果
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        #cv2.imshow('res',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
'''
函数：FaceDetectFromDir（）
函数功能：批量检测人脸
输入参数：indir----输入文件夹
         savedir----保存文件
'''
def FaceDetectFromDir(indir,savedir,new_w,new_h):
    if not os.path.exists(indir):
        print u'输入路径不存在'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dirs = os.listdir(indir)
    for sub in dirs:
        outdir = savedir+'/'+sub
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
        subdir = indir+'/'+sub
        files = os.listdir(subdir)
        for fr in files:
            filename = subdir+'/'+fr
            outname = outdir + '/' + fr
            FaceDetect(filename,outname,new_w,new_h)

'''
函数：FaceDetectS（）
函数功能：批量检测人脸
输入参数：srcPath----输入文件夹
         dstPath----输出文件夹
'''
def FaceDetectS(srcPath,dstPath,filelist='imageBbox_detect_replace.list',w=128,h=128):
    if not os.path.exists(srcPath):
        print u'输入路径不存在'
    if not os.path.exists(dstPath):
        os.makedirs(savedir)
    fid=open(filelist)
    lines=fid.readlines()
    fid.close()
    for line in lines:
        word=line.split()
        filename=word[0]
        iname = srcPath+filename
        savename=dstPath+filename
        dirname, basename = os.path.split(savename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        FaceDetect(iname,savename,w,h)
if __name__=='__main__':
    indir = r'F:/Small_data/aligned'
    savedir=r'F:/Small_data/croped'
    FaceDetectFromDir(indir,savedir,144,144)