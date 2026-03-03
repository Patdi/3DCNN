import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import bisect
import math
import scipy.ndimage
from layers import *
import random
from atom_res_dict import *
import argparse
from theano.misc.pkl_utils import load

def load_weights_pickle(file_name,dropout_rate):
	keys=numpy.load(file_name).keys()
	W0=numpy.load(file_name)[keys[0]]
	W1=numpy.load(file_name)[keys[1]]
	W2=numpy.load(file_name)[keys[2]]
	W3=numpy.load(file_name)[keys[3]]
	W4=numpy.load(file_name)[keys[4]]
	W5=numpy.load(file_name)[keys[5]]
	b0=numpy.load(file_name)[keys[6]]
	b1=numpy.load(file_name)[keys[7]]
	b2=numpy.load(file_name)[keys[8]]
	b3=numpy.load(file_name)[keys[9]]
	b4=numpy.load(file_name)[keys[10]]
	b5=numpy.load(file_name)[keys[11]]

	W0=theano.shared(value=W0*(1 - dropout_rate), name='W0', borrow=True)
	W1=theano.shared(value=W1*(1 - dropout_rate), name='W1', borrow=True)
	W2=theano.shared(value=W2*(1 - dropout_rate), name='W2', borrow=True)
	W3=theano.shared(value=W3*(1 - dropout_rate), name='W3', borrow=True)
	W4=theano.shared(value=W4*(1 - dropout_rate), name='W4', borrow=True)
	W5=theano.shared(value=W5, name='W5', borrow=True)

	b0=theano.shared(value=b0*(1 - dropout_rate), name='b0', borrow=True)
	b1=theano.shared(value=b1*(1 - dropout_rate), name='b1', borrow=True)
	b2=theano.shared(value=b2*(1 - dropout_rate), name='b2', borrow=True)
	b3=theano.shared(value=b3*(1 - dropout_rate), name='b3', borrow=True)
	b4=theano.shared(value=b4*(1 - dropout_rate), name='b4', borrow=True)
	b5=theano.shared(value=b5, name='b5', borrow=True)
	return [W0,W1,W2,W3,W4,W5,b0,b1,b2,b3,b4,b5]



def shared_dataset(data_x, data_y, borrow=True):

		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		return shared_x, T.cast(shared_y, 'int32')

def load_box(num):
	mean = numpy.load("../data/Sampled_Numpy/train/train_mean.dat") 
	Xt = numpy.load("../data/Sampled_Numpy/test/Xt_smooth_"+str(num+1)+".dat")
	yt = numpy.load("../data/Sampled_Numpy/test/yt_"+str(num+1)+".dat")
	Xt -= mean 
	print ("test X size: "+ str(Xt.shape[0]))
	test_set_x, test_set_y = shared_dataset(Xt, yt)
	test_set_x=test_set_x.dimshuffle(0,4,1,2,3)
   
	datasets = [test_set_x, test_set_y]

	return datasets
		
if __name__ == '__main__':
	weights_ID = '3DCNN_backbone'
	dropout_rate = 0.3
	num_3d_pixel = 20
	flt_channels  = 100
	num_of_parts = 6

	[W0,W1,W2,W3,W4,W5,b0,b1,b2,b3,b4,b5] = load_weights_pickle(file_name='../weights/weight_'+weights_ID+'.zip',dropout_rate=dropout_rate)

	for num in range(0,num_of_parts):

		datasets=load_box(num)
		[test_set_x, test_set_y] = datasets

		total_size = test_set_y.eval().shape[0]
		batch_size = 100
		num_of_test_parts = total_size/batch_size


		for r in range(0,num_of_test_parts):
			if batch_size*(r+1)>total_size:
				x=test_set_x[batch_size*r:total_size]
				y=test_set_y[batch_size*r:total_size]
			else:
				x=test_set_x[batch_size*r:batch_size*(r+1)]
				y=test_set_y[batch_size*r:batch_size*(r+1)]


			batch_size=y.shape.eval()[0]
			

			rng = numpy.random.RandomState(23455)

			in_channels = 4
			filter_w = 3

			######################
			# BUILD ACTUAL MODEL #
			######################
			print ('... building the model')
			# image sizes
			batchsize     = batch_size
			in_time       = num_3d_pixel
			in_width      = num_3d_pixel
			in_height     = num_3d_pixel
			#filter sizes
			flt_channels  = 100
			flt_time      = filter_w
			flt_width     = filter_w
			flt_height    = filter_w

			layer0_w = num_3d_pixel
			layer0_h = num_3d_pixel
			layer0_d = num_3d_pixel

			# ====== net1 =======

			layer1_w = (layer0_w-3+1) #14
			layer1_h = (layer0_h-3+1)
			layer1_d = (layer0_d-3+1)

			layer2_w = (layer1_w-3+1)/2 #14
			layer2_h = (layer1_h-3+1)/2
			layer2_d = (layer1_d-3+1)/2

			layer3_w = (layer2_w-3+1)/2
			layer3_h = (layer2_h-3+1)/2
			layer3_d = (layer2_d-3+1)/2

			signals_shape0 = (batchsize, in_time, in_channels, in_height, in_width)
			filters_shape0 = (flt_channels, 3, in_channels, 3, 3)
			signals_shape1 = (batchsize, layer1_d, flt_channels, layer1_h, layer1_w)
			filters_shape1 = (flt_channels*2, 3, flt_channels, 3, 3)
			signals_shape2 = (batchsize, layer2_d, flt_channels*2, layer2_h, layer2_w)
			filters_shape2 = (flt_channels*4, 3, flt_channels*2, 3, 3)

				
			layer0_input = x.reshape(signals_shape0) #20

			layer0 = Conv_3d_Layer(rng, input=layer0_input, #18
					image_shape=signals_shape0,
					filter_shape=filters_shape0, W=W0, b=b0)

			layer1 = Conv_3d_Layer(rng, input=layer0.output, #8
					image_shape=signals_shape1,
					filter_shape=filters_shape1, W=W1, b=b1)

			layer1_pool = PoolLayer3D(input=layer1.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

			layer2 = Conv_3d_Layer(rng, input=layer1_pool.output.dimshuffle(0,2,1,3,4), 
					image_shape=signals_shape2,
					filter_shape=filters_shape2, W=W2, b=b2)

			layer2_pool = PoolLayer3D(input=layer2.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

			layer3_input = layer2_pool.output.dimshuffle(0,2,1,3,4).flatten(2) 

			layer3 = HiddenLayer(rng, input=layer3_input, n_in=(flt_channels*4*layer3_d*layer3_w*layer3_h), 
								 n_out=1000, activation=relu, W=W3, b=b3)
			layer4 = HiddenLayer(rng, input=layer3.output, n_in=1000, 
								 n_out=100, activation=relu, W=W4, b=b4)

			layer5 = LogisticRegression(input=layer4.output, n_in=100, n_out=20, W=W5, b=b5) 

			pred = layer5.y_pred.eval()


			print ("correct labels")
			print (y.eval())
			print ("predicted labels")
			print (pred)
			

			y_true=numpy.array(y.eval())
			y_pred=numpy.array(pred)

			if num==0 and r == 0:
				all_y_true = y_true
				all_y_pred = y_pred
			else:
				all_y_true = numpy.concatenate((all_y_true,y_true),axis=0)
				all_y_pred = numpy.concatenate((all_y_pred,y_pred),axis=0)

		all_y_true.dump('../results/labels_'+str(num+1)+'.dat')
		all_y_pred.dump('../results/pred_'+str(num+1)+'.dat')

		acc_group=0
		correct=0
		for i in range (0,all_y_true.shape[0]):
			if(int(all_y_true[i])==int(all_y_pred[i])):
				correct=correct+1
			if res_group_dict[int(all_y_true[i])]==res_group_dict[int(all_y_pred[i])]:
				acc_group=acc_group+1
		
		correct=float(correct)/all_y_true.shape[0]
		acc_group=float(acc_group)/all_y_true.shape[0]
		
		print ("acc_group:")
		print (acc_group)
		print ("individual accuracy:")
		print (correct)

   