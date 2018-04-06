import sys
pycaffePath = '~/software/caffe/python' 
sys.path.append(pycaffePath)
import caffe
import numpy as np
import random
import cPickle as pickle
imdb_exit = True
#####修改参数#####
clothes_class = 'blouse' 
point_number = 24  #点个数
trainImdb = '../data/IMDB/' + clothes_class+ ('/%s_train.imdb'%clothes_class)
testImdb = '../data/IMDB/' + clothes_class+ ('/%s_test.imdb'%clothes_class)
#########################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_Train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 64
	    net_side = 224  #net
        self.batch_loader = BatchLoader(trainImdb, net_side)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size,point_number*2)
        top[2].reshape(self.batch_size,point_number*2)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for index in range(self.batch_size):
            im, reg, valid_value = self.batch_loader.load_next_image()
            top[0].data[index, ...] = im
            top[1].data[index, ...] = reg
            top[2].data[index, ...] = valid_value
    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    def __init__(self, fileName, net_side):
        self.data_list = []
        self.mean = np.array([100,100,100]).reshape(1,1,3) #平均值
        self.count = 0 
        self.image_size = net_side    
        print "start Reading Classify into memory ..."
        file = open(fileName, 'r')  
        self.data_list = pickle.load(file)
        file.close()
        print str(len(self.data_list)),"success load memory"

   
    def load_next_image(self):
        if self.count == len(self.data_list):
            self.count = 0
        count_data = self.data_list[self.count]

        img = np.array(count_data[0], dtype= float)
        #add 
        img = np.swapaxes(img, 0, 2)
        img = img - self.mean
        reg = np.array(count_data[1], dtype = float) / self.image_size
        valid_value = np.array(count_data[2], dtype = int)            
        self.count += 1
        #print label
        return img, reg, valid_value
################################################################################
#########################landMark Loss Layer By Python###############################
################################################################################
class Regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
		if len(bottom) != 3:
		    raise Exception("Need 3 Inputs")
    def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count or bottom[2].count != bottom[0].count or bottom[2].count != bottom[1].count :
		    raise Exception("Input predict and groundTruth should have same dimension")
		pointClass = bottom[2].data
		self.valid_index = np.where(pointClass != -1)[0]
	    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
	    top[0].reshape(1)

    def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0
		labelPoint = bottom[1].data
		predictPoint = np.array(bottom[0].data).reshape(bottom[1].data.shape)
		labelPoint[self.valid_index] = 0
		predictPoint[self.valid_index] = 0
		predictPoint = np.array(predictPoint).reshape(bottom[0].data.shape)
		self.diff[...] = predictPoint - np.array(labelPoint).reshape(bottom[0].data.shape)
	    top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
		for i in range(3):
		    if not propagate_down[i]:
			   continue
		    if i == 0:
			   sign = 1
		    else:
			   sign = -1
		    bottom[i].diff[...] = sign * self.diff / bottom[i].num

class Data_Layer_Test(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 128
	    net_side = 224  #输入网络大小
        self.batch_loader = BatchLoader(testImdb, net_side)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size,point_number*2)
        top[2].reshape(self.batch_size,point_number*2)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for index in range(self.batch_size):
            im, reg, valid_value = self.batch_loader.load_next_image()
            top[0].data[index, ...] = im
            top[1].data[index, ...] = reg
            top[2].data[index, ...] = valid_value
    def backward(self, top, propagate_down, bottom):
        pass
class Test_Layer_Loss(caffe.Layer):
    def setup(self,bottom,top):
		if len(bottom) != 3:
		    raise Exception("Need 3 Inputs")
    def reshape(self,bottom,top):
		if bottom[0].count != bottom[1].count or bottom[2].count != bottom[0].count || bottom[2].count != bottom[1].count :
		    raise Exception("Input predict and groundTruth should have same dimension")
		pointClass = bottom[2].data
	    self.valid_index = np.where(pointClass != -1)[0]
	    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
	    top[0].reshape(1)

    def forward(self,bottom,top):
		self.diff[...] = 0
		top[0].data[...] = 0
		labelPoint = bottom[1].data
		predictPoint = np.array(bottom[0].data).reshape(bottom[1].data.shape)
		labelPoint[self.valid_index] = 0
		predictPoint[self.valid_index] = 0
		predictPoint = np.array(predictPoint).reshape(bottom[0].data.shape)
		self.diff[...] = predictPoint - np.array(labelPoint).reshape(bottom[0].data.shape)
	    top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
		pass