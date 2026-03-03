from __future__ import division;
from __future__ import print_function;
import tables;
import numpy
import sys


DEFAULT_NODE_NAME = "defaultNode";

def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
    node = h5file.get_node('/', DEFAULT_NODE_NAME);
    return getattr(node, columnName);

def load_ATOM_BOX(num_of_parts):
	
	dataName = "data";
	dataShape = [4,20,20,20]; #arr describing the dimensions other than the extendable dim.
	labelName = "label";
	labelShape = [];

	all_Xtr=[]
	all_ytr=[]
	all_train_sizes=[]
	train_mean=numpy.zeros((4,20,20,20))
	total_train_size=0

	for part in range (0,num_of_parts):

		filename_train = "../data/ATOM_CHANNEL_dataset/train_data_"+str(part+1)+".pytables";
		h5file_train = tables.open_file(filename_train, mode="r")
		dataColumn_train = getH5column(h5file_train, dataName);
		labelColumn_train = getH5column(h5file_train, labelName);
		Xtr=dataColumn_train[:]
		ytr=labelColumn_train[:]
		total_train_size+=Xtr.shape[0]
		
		all_train_sizes.append(Xtr.shape[0])
		all_Xtr.append(Xtr)
		all_ytr.append(ytr)

	
	mean = numpy.load("../data/Sampled_Numpy/train/train_mean.dat")
	norm_Xtr = []
	for Xtr in all_Xtr:
		Xtr -= mean
		norm_Xtr.append(Xtr)


	# Due to memorry consideration and training speed, we only used 1/6 val data to get a sense of the general val error. 
	for part in range (0,1):
		filename_val = "../data/ATOM_CHANNEL_dataset/val_data_"+str(part+1)+".pytables";
		h5file_val = tables.open_file(filename_val, mode="r")
		dataColumn_val = getH5column(h5file_val, dataName);
		labelColumn_val = getH5column(h5file_val, labelName);
		Xv=dataColumn_val[:]
		yv=labelColumn_val[:]
		Xv -= mean
		
		if part == 0:
			norm_Xv = Xv
			all_yv = yv
		else:
			norm_Xv = numpy.concatenate((norm_Xv,Xv), axis=0)
			all_yv = numpy.concatenate((all_yv,yv), axis=0)

	all_examples=[norm_Xtr,norm_Xv]
	all_labels=[all_ytr,all_yv]
	return [all_examples, all_labels, all_train_sizes, norm_Xv.shape[0]]



	

