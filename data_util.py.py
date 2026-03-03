from __future__ import division;
from __future__ import print_function;
import tables;
import numpy
import sys


DEFAULT_NODE_NAME = "defaultNode";


def init_h5_file(toDiskName, groupName=DEFAULT_NODE_NAME, groupDescription=DEFAULT_NODE_NAME):
	"""
		toDiskName: the name of the file on disk
	"""
	import tables;
	h5file = tables.openFile(toDiskName, mode="w", title="Dataset")
	gcolumns = h5file.createGroup(h5file.root, groupName, groupDescription)
	return h5file;

class InfoToInitArrayOnH5File(object):
	def __init__(self, name, shape, atomicType):
		"""
			name: the name of this matrix
			shape: tuple indicating the shape of the matrix (similar to numpy shapes)
			atomicType: one of the pytables atomic types - eg: tables.Float32Atom() or tables.StringAtom(itemsize=length);
		"""
		self.name = name;
		self.shape = shape;
		self.atomicType = atomicType;

def writeToDisk(theH5Column, whatToWrite, batch_size=5000):
	"""
		Going to write to disk in batches of batch_size
	""" 
	data_size = len(whatToWrite);
	last = int(data_size / float(batch_size)) * batch_size
	for i in xrange(0, data_size, batch_size):
		stop = (i + data_size%batch_size if i >= last
				else i + batch_size)
		theH5Column.append(whatToWrite[i:stop]);
		h5file.flush()
	
def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
	node = h5file.getNode('/', DEFAULT_NODE_NAME);
	return getattr(node, columnName);


def initColumnsOnH5File(h5file, infoToInitArraysOnH5File, expectedRows, nodeName=DEFAULT_NODE_NAME, complib='blosc', complevel=5):
	"""
		h5file: filehandle to the h5file, initialised with init_h5_file
		infoToInitArrayOnH5File: array of instances of InfoToInitArrayOnH5File
		expectedRows: this code is set up to work with EArrays, which can be extended after creation.
			(presumably, if your data is too big to fit in memory, you're going to have to use EArrays
			to write it in pieces). "sizeEstimate" is the estimated size of the final array; it
			is used by the compression algorithm and can have a significant impace on performance.
		nodeName: the name of the node being written to.
		complib: the docs seem to recommend blosc for compression...
		complevel: compression level. Not really sure how much of a difference this number makes...
	"""
	gcolumns = h5file.getNode(h5file.root, nodeName);
	filters = tables.Filters(complib=complib, complevel=complevel);
	for infoToInitArrayOnH5File in infoToInitArraysOnH5File:
		finalShape = [0]; #in an eArray, the extendable dimension is set to have len 0
		finalShape.extend(infoToInitArrayOnH5File.shape);
		h5file.createEArray(gcolumns, infoToInitArrayOnH5File.name, atom=infoToInitArrayOnH5File.atomicType
							, shape=finalShape, title=infoToInitArrayOnH5File.name #idk what title does...
							, filters=filters, expectedrows=expectedRows);



def load_ATOM_BOX():
	
	dataName = "data";
	dataShape = [4,20,20,20]; #arr describing the dimensions other than the extendable dim.
	labelName = "label";
	labelShape = [];

	all_Xtr=[]
	all_ytr=[]
	all_train_sizes=[]
	train_mean=numpy.zeros((4,20,20,20))
	total_train_size=0

	for part in range (0,6):

		filename_train = "../data/ATOM_CHANNEL_dataset/train_data_"+str(part+1)+".pytables";
		h5file_train = tables.openFile(filename_train, mode="r")
		dataColumn_train = getH5column(h5file_train, dataName);
		labelColumn_train = getH5column(h5file_train, labelName);
		Xtr=dataColumn_train[:]
		ytr=labelColumn_train[:]
		total_train_size+=Xtr.shape[0]
		train_mean += numpy.mean(Xtr, axis=0)
		
		all_train_sizes.append(Xtr.shape[0])
		all_Xtr.append(Xtr)
		all_ytr.append(ytr)

	
	mean = train_mean/6
	norm_Xtr = []
	for Xtr in all_Xtr:
		Xtr -= mean
		norm_Xtr.append(Xtr)


	# Due to memorry consideration and training speed, we only used 1/6 test data to get a sense of the general test error. 
	# We test the full test dataset separately after the training is completed.
	for part in range (0,1): 
		filename_test = "../data/ATOM_CHANNEL_dataset/test_data_"+str(part+1)+".pytables";
		h5file_test = tables.openFile(filename_test, mode="r")
		dataColumn_test = getH5column(h5file_test, dataName);
		labelColumn_test = getH5column(h5file_test, labelName);
		Xt=dataColumn_test[:]
		yt=labelColumn_test[:]
		Xt -= mean 

		if part == 0:
			norm_Xt = Xt
			all_yt = yt
		else:
			norm_Xt = numpy.concatenate((norm_Xt,Xt), axis=0)
			all_yt = numpy.concatenate((all_yt,yt), axis=0)
	   
	# Same considerations as the above for the test dataset, more val data can be used to tune the hyper-parameters if desired
	for part in range (0,1):
		filename_val = "../data/ATOM_CHANNEL_dataset/val_data_"+str(part+1)+".pytables";
		h5file_val = tables.openFile(filename_val, mode="r")
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

	all_examples=[norm_Xtr,norm_Xt,norm_Xv]
	all_labels=[all_ytr,all_yt,all_yv]
	return [all_examples, all_labels, all_train_sizes, norm_Xt.shape[0], norm_Xv.shape[0]]


def load_FEATURE():
	ID = 'FEATURE_SCOP_T4_train_beta'
	for part in range(0,5):
		data_part = numpy.load("../data/FEATURE_dataset/FEATURE_train_X_"+str(part)+".dat")
		labels_part = numpy.load("../data/FEATURE_dataset/FEATURE_train_y_"+str(part)+".dat")
		if part ==0:
			data = data_part
			labels = labels_part
		else:
			data = numpy.concatenate((data,data_part),axis=0)
			labels = numpy.concatenate((labels,labels_part),axis=0)
	data_mean = numpy.mean(data,axis=0)
	Xv = numpy.load("../data/FEATURE_dataset/FEATURE_val_X.dat")
	yv = numpy.load("../data/FEATURE_dataset/FEATURE_val_y.dat")

	Xt = numpy.load("../data/FEATURE_dataset/FEATURE_test_X.dat")
	yt = numpy.load("../data/FEATURE_dataset/FEATURE_test_y.dat")

	Xv -= data_mean
	Xt -= data_mean
	data -= data_mean

	Xtr=data
	ytr=labels

	return [Xtr, ytr, Xt, yt, Xv, yv]
	

