# -- coding: utf-8 --
import sys
import cv2 as cv
import os
import numpy as np
import cPickle as pickle
import pandas as pd

bgr=True

width=224
length=224

train=6 #数据集的划分
test=2
val=2

Path_0='IMDB/' #确定你要生成的IMDB文件存放的主文件夹


total=train+test+val
blouse_list = []  #最终要写入IMDB的List
dress_list = []
outwear_list = []
skirt_list = []
trousers_list = []


def takeImage(i,category,Imagelist,data):
	if category!=data.ix[i,1]:#判断是否和Imagelist是同一个所属类别的数据，不是则返回
		return;
	if bgr==True:
		img_0 = cv.imread(imgWay+data.ix[i,0],1) #读取图像
	else:
		img_0 = cv.imread(imgWay+data.ix[i,0],0) #读取图像
	img_1=np.array(img_0)
	img = cv.resize(img_0,(width,length))#获得指定大小的图像
	img_d=cv.flip(img, 1)
	temp=[]
	temp_d=[]
	temp.append(img)
	temp_d.append(img_d) #添加图像翻转
	start=2#跳过图片的地址和类别
	for j in range(len(data.ix[i])-2):
		point = data.ix[i,start].split('_',2)
		x=int(point[0])*width/img_1.shape[1]
		y=int(point[1])*length/img_1.shape[0]
		visual=int(point[2])
		temp.append([x,y,visual])
		temp_d.append([width-x,y,visual])
		start=start+1
	Imagelist.append(temp)#分别写入同一张图片以及它的镜像的数据，在后面会对所有数据进行打乱
	Imagelist.append(temp_d)

def makeList(annotations): #将指定annotations文件的数据生成IMDB
	fp = open(annotations,'r')
	data = pd.read_csv(fp)
	sum_ = len(data)#确定要写入前几张图片的数据
	for i in range(100):
		takeImage(i,'blouse',blouse_list,data)
		takeImage(i,'dress',dress_list,data)
		takeImage(i,'outwear',outwear_list,data)
		takeImage(i,'skirt',skirt_list,data)
		takeImage(i,'trousers',trousers_list,data)
	fp.close()


def saveImdb():#将ImageList分别以train、test和val写入IMDB文件，采取的做法是先打乱，再按比例对ImageList分区间写入IMDB
	endName='.imdb'
	saveWay=[Path_0+'blouse/blouse',Path_0+'dress/dress',Path_0+'outwear/outwear',Path_0+'skirt/skirt',Path_0+'trousers/trousers']
	Imagelist=[blouse_list,dress_list,outwear_list,skirt_list,trousers_list]
	for i in range(len(Imagelist)):
		np.random.shuffle(Imagelist[i])
		fid_train = open(saveWay[i]+'_train'+endName,'w')
		fid_test = open(saveWay[i]+'_test'+endName,'w')
		fid_val = open(saveWay[i]+'_val'+endName,'w')

		pickle.dump(Imagelist[i][0:len(Imagelist[i])*train/total], fid_train)
		pickle.dump(Imagelist[i][len(Imagelist[i])*train/total:len(Imagelist[i])*(test+train)/total], fid_test)
		pickle.dump(Imagelist[i][len(Imagelist[i])*(test+train)/total:len(Imagelist[i])], fid_val)

		fid_train.close()
		fid_test.close()
		fid_val.close()

def makeDir():#创建指定文件夹目录
	Dir=['blouse','dress','outwear','skirt','trousers']
	for i in range(len(Dir)):
		if os.path.exists(Path_0+Dir[i]) == True:
			continue
		os.makedirs(Path_0+Dir[i])

makeDir()
imgWay='train_1/' #该训练集Image文件夹的目录位置
makeList('train_1/Annotations/annotations.csv')

imgWay='train_2/'
makeList('train_2/Annotations/train.csv')

saveImdb()#将生成的imdb文件保存在当前的目录下面