# from __future__ import division;
# from __future__ import print_function;
# import tables;
import numpy
import os
import random
import json
import scipy.ndimage
from scipy import spatial
import sys

DEFAULT_NODE_NAME = "defaultNode";

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}

resiName_to_label={'ILE': 12, 'GLN': 8, 'GLY': 18, 'GLU': 4, 'CYS': 19, 'HIS': 0, 'SER': 5, 'LYS': 1, 'PRO': 17, 'ASN': 7, 'VAL': 10, 'THR': 6, 'ASP': 3, 'TRP': 16, 'PHE': 14, 'ALA': 9, 'MET': 13, 'LEU': 11, 'ARG': 2, 'TYR': 15}

def load_dict(dict_name):
	if os.path.isfile(os.path.join('../data/DICT',dict_name)):
		with open(os.path.join('../data/DICT',dict_name)) as f:
			tmp_dict = json.load(f)
		res_count_dict={}
		for i in range (0,20):
			res_count_dict[i]=tmp_dict[str(i)]
	else:
		print ("dictionary not exist!")
		res_count_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0}
		
		files = [ f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir,f))]
		for f in files:
			file_name=f.strip('\n')
			parts=file_name.split('_')
			res=parts[0]
			label=resiName_to_label[res]
			res_count_dict[label]+=1
	
	print ("res_count_dict content:")
	for key in res_count_dict:
		print (label_res_dict[key]+" "+str(res_count_dict[key]))

	min_ind = min(res_count_dict, key=res_count_dict.get)
	min_data=res_count_dict[min_ind]
	print (min_ind,min_data)

	res_files_dict={}
	for label in range (0,20):
		mask=random.sample(range(res_count_dict[label]), min_data)
		res_files_dict[label]=mask

	return res_count_dict, res_files_dict, min_ind, min_data


def integrate_data(d_name, num_3d_pixel=20, num_of_channels=4, num_of_parts=6):

	print d_name
	block_size = 1000 if d_name == 'train' else 100
	dict_name = d_name+'_20AA_boxes.json'
	in_dir = '../data/RAW_DATA/'+d_name+'/'
	out_dir = '../data/Sampled_Numpy/'+d_name+'/'

	res_count_dict, res_files_dict, min_ind, min_data = load_dict(dict_name)
	unit_size = int(20*(min_data/num_of_parts)*block_size)

	for part in range(0,num_of_parts):
		equal_examples=[]
		equal_labels=[]
		for label in range (0,20):
			res_files = res_files_dict[label]
			s = int(part*(min_data/num_of_parts))
			e = int((part+1)*(min_data/num_of_parts))
			for i in range (s,e):
				num = res_files[i]
				X = numpy.load(in_dir+label_res_dict[label]+"_"+str(num)+'.dat')
				y=label*numpy.ones((block_size,1))
				equal_examples.append(X)
				equal_labels.append(y)

		equal_examples=numpy.array(equal_examples)
		equal_labels=numpy.array(equal_labels)

		print "equal_examples.shape"
		print "equal_labels.shape"
		print equal_examples.shape
		print equal_labels.shape

		equal_examples=numpy.reshape(equal_examples,(unit_size, num_of_channels, num_3d_pixel, num_3d_pixel, num_3d_pixel))
		equal_labels=numpy.reshape(equal_labels,unit_size)

		print "equal_examples.shape"
		print "equal_labels.shape"
		print equal_examples.shape
		print equal_labels.shape

		if d_name=='test':
			Xt_smooth=equal_examples
			yt=equal_labels
			Xt_smooth.dump(out_dir+"Xt_smooth_"+str(part+1)+".dat")
			yt.dump(out_dir+"yt_"+str(part+1)+".dat")

			print "Xt_smooth.shape"
			print "yt.shape"
			print Xt_smooth.shape
			print yt.shape

		else:

			num_of_train=int(19*float(unit_size)/20)
			num_of_val=int(1*float(unit_size)/20)

			mask_train=random.sample(xrange(unit_size), num_of_train)
			X_smooth=equal_examples[mask_train]
			y=equal_labels[mask_train]
			equal_examples=numpy.delete(equal_examples, mask_train, 0)
			equal_labels=numpy.delete(equal_labels, mask_train, 0)

			Xv_smooth=equal_examples
			yv=equal_labels

			# Dumping validation dataset as numpy array
			Xv_smooth.dump(out_dir+"Xv_smooth_"+str(part+1)+".dat")
			yv.dump(out_dir+"yv_"+str(part+1)+".dat")

			print "Xv_smooth.shape"
			print "yv.shape"
			print Xv_smooth.shape
			print yv.shape

			
			train_mean = numpy.mean(X_smooth, axis=0)
			train_mean.dump("../data/Sampled_Numpy/train/train_mean.dat")


			# Dumping training dataset as numpy array
			partition = int(num_of_train/19)
			for i in range (1,20):
				mask = range(partition*(i-1),partition*i)
				X_tmp = X_smooth[mask]
				y_tmp = y[mask]
				print "X_tmp.shape"
				print "y_tmp.shape"
				print X_tmp.shape
				print y_tmp.shape
				X_tmp.dump(out_dir+"X_smooth"+str(i)+"_"+str(part+1)+".dat")
				y_tmp.dump(out_dir+"y"+str(i)+"_"+str(part+1)+".dat")

				


if __name__ == '__main__':
	
	num_of_channels=4
	num_3d_pixel=20

	# integrate training and validation data
	integrate_data('train', num_3d_pixel, num_of_channels, num_of_parts=6)
	# integrate test data
	integrate_data('test', num_3d_pixel, num_of_channels, num_of_parts=6)

