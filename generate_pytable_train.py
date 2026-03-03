#!/usr/bin/env python
from __future__ import division;
from __future__ import print_function;
import tables;
import numpy
import sys

DEFAULT_NODE_NAME = "defaultNode";


def init_h5_file(toDiskName, groupName=DEFAULT_NODE_NAME, groupDescription=DEFAULT_NODE_NAME):

	h5file = tables.open_file(toDiskName, mode="w", title="Dataset")
	gcolumns = h5file.create_group(h5file.root, groupName, groupDescription)
	return h5file;

class InfoToInitArrayOnH5File(object):
	def __init__(self, name, shape, atomicType):

		self.name = name;
		self.shape = shape;
		self.atomicType = atomicType;

def writeToDisk(h5file,theH5Column, whatToWrite, batch_size=5000):

	data_size = len(whatToWrite);
	last = int(data_size / float(batch_size)) * batch_size
	for i in xrange(0, data_size, batch_size):
		stop = (i + data_size%batch_size if i >= last
				else i + batch_size)
		theH5Column.append(whatToWrite[i:stop]);
		h5file.flush()
	
def getH5column(h5file, columnName, nodeName=DEFAULT_NODE_NAME):
	node = h5file.get_node('/', DEFAULT_NODE_NAME);
	return getattr(node, columnName);


def initColumnsOnH5File(h5file, infoToInitArraysOnH5File, expectedRows, nodeName=DEFAULT_NODE_NAME, complib='blosc', complevel=5):
	gcolumns = h5file.get_node(h5file.root, nodeName);
	filters = tables.Filters(complib=complib, complevel=complevel);
	for infoToInitArrayOnH5File in infoToInitArraysOnH5File:
		finalShape = [0]; #in an eArray, the extendable dimension is set to have len 0
		finalShape.extend(infoToInitArrayOnH5File.shape);
		h5file.create_earray(gcolumns, infoToInitArrayOnH5File.name, atom=infoToInitArrayOnH5File.atomicType
							, shape=finalShape, title=infoToInitArrayOnH5File.name #idk what title does...
							, filters=filters, expectedrows=expectedRows);

	
if __name__ == "__main__":

	d_set = 'train'
	num_of_channels=4
	num_of_parts = 6
	num_3d_pixel=20
	
	#intiialise the columns going on the file
	dataName = "data";
	dataShape = [num_of_channels,num_3d_pixel,num_3d_pixel,num_3d_pixel]; #arr describing the dimensions other than the extendable dim.
	labelName = "label";
	labelShape = []; #the outcome is a vector, so there's only one dimension, the extendable one.
	dataInfo = InfoToInitArrayOnH5File(dataName, dataShape, tables.Float32Atom());
	labelInfo = InfoToInitArrayOnH5File(labelName, labelShape, tables.Float32Atom());

	for part in range (1,num_of_parts+1):
		Xv_smooth = numpy.load("../data/Sampled_Numpy/"+d_set+"/Xv_smooth_"+str(part)+".dat")
		yv = numpy.load("../data/Sampled_Numpy/"+d_set+"/yv_"+str(part)+".dat")

		for i in range (1,20):
			X = numpy.load("../data/Sampled_Numpy/"+d_set+"/X_smooth"+str(i)+'_'+str(part)+".dat")
			y = numpy.load("../data/Sampled_Numpy/"+d_set+"/y"+str(i)+"_"+str(part)+".dat")
			
			if i==1:
				X_smooth = X
				labels = y[:,numpy.newaxis]
			else:
				X_smooth = numpy.concatenate((X_smooth,X), axis=0)
				labels = numpy.concatenate((labels,y[:,numpy.newaxis]), axis=0)

		labels = numpy.ravel(labels)

		# Writing Train pytables
		filename_train = "../data/ATOM_CHANNEL_dataset/train_data_"+str(part)+".pytables";
		h5file = init_h5_file(filename_train);
		numSamples = X_smooth.shape[0];
		
		initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples);
		dataColumn = getH5column(h5file, dataName);
		labelColumn = getH5column(h5file, labelName); 
		writeToDisk(h5file, dataColumn, X_smooth);
		writeToDisk(h5file, labelColumn, labels);
		h5file.close();

		# Writing Val pytables
		filename_val = "../data/ATOM_CHANNEL_dataset/val_data_"+str(part)+".pytables"; 
		h5file = init_h5_file(filename_val);
		numSamples = Xv_smooth.shape[0];
		
		initColumnsOnH5File(h5file, [dataInfo,labelInfo], numSamples);
		dataColumn = getH5column(h5file, dataName);
		labelColumn = getH5column(h5file, labelName); 
		writeToDisk(h5file,dataColumn, Xv_smooth);
		writeToDisk(h5file,labelColumn, yv);
		h5file.close();
