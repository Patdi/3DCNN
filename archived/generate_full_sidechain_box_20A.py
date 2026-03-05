import math
import numpy
import os
import sys
import theano
import theano.tensor as T
import scipy
from scipy import spatial
import json
import collections
import argparse
from atom_res_dict import *

GLY=[]
CYS=[]
ARG=[]
SER=[]
THR=[]
LYS=[]
MET=[]
ALA=[]
LEU=[]
ILE=[]
VAL=[]
ASP=[]
GLU=[]
HIS=[]
ASN=[]
PRO=[]
GLN=[]
PHE=[]
TRP=[]
TYR=[]

res_container_dict={0:HIS,1:LYS,2:ARG,3:ASP,4:GLU,5:SER,6:THR,7:ASN,8:GLN,9:ALA,10:VAL,11:LEU,12:ILE,13:MET,14:PHE,15:TYR,16:TRP,17:PRO,18:GLY,19:CYS}

class PDB_atom:
	def __init__(self,atom_type,res,chain_ID,x,y,z,index,value):
		self.atom = atom_type
		self.res = res
		self.chain_ID = chain_ID
		self.x = x
		self.y = y
		self.z = z
		self.index = index
		self.value = value
	def __eq__(self, other): 
		return self.__dict__ == other.__dict__

def parse_processed_list(name_list):
	exist_PDB = Set([])
	for l in name_list:
		list_file= open(l)
		f = list(list_file)
		for line in f:
			PDB_ID=line.split()[-1]
			exist_PDB.add(PDB_ID)
	return exist_PDB

def center_and_transform(label,get_position):

	reference = get_position["CA"]
	axis_x = numpy.array(get_position["N"]) - numpy.array(get_position["CA"])  
	pseudo_axis_y = numpy.array(get_position["C"]) - numpy.array(get_position["CA"])  
	axis_z = numpy.cross(axis_x , pseudo_axis_y)
	if not label==18:
		direction = numpy.array(get_position["CB"]) - numpy.array(get_position["CA"]) 
		axis_z *= numpy.sign( direction.dot(axis_z) ) 
	axis_y= numpy.cross(axis_z , axis_x)

	axis_x/=numpy.sqrt(sum(axis_x**2))
	axis_y/=numpy.sqrt(sum(axis_y**2))
	axis_z/=numpy.sqrt(sum(axis_z**2))

	transform=numpy.array([axis_x, axis_y, axis_z], 'float16').T
	return [reference,transform]

def dist(cor1,cor2):
	return math.sqrt((cor1[0]-cor2[0])**2+(cor1[1]-cor2[1])**2+(cor1[2]-cor2[2])**2)

def find_actual_pos(my_kd_tree,cor,PDB_entries):
	[d,i] = my_kd_tree.query(cor,k=1)
	return PDB_entries[i]

def get_position_dict(all_PDB_atoms):
	get_position={}
	for a in all_PDB_atoms:
		get_position[a.atom]=(a.x,a.y,a.z)
	return get_position

def load_pdb_set(d_name):
	PDB_list_file = open('../data/PDB_'+d_name+'.txt')
	pdb_dir = '../data/PDB_family_'+d_name
	PDB_Set = set()
	for line in PDB_list_file:
		PDB_ID = line.split()[0]
		PDB_Set.add(PDB_ID.lower())
	return PDB_Set


def grab_PDB(entry_list):
	ID_dict=collections.OrderedDict()
	all_pos=[]
	all_lines=[]
	all_atom_type =[]
	PDB_entries = []
	atom_index = 0
	model_ID = 0
	MODELS = []
	all_x = []
	all_y = []
	all_z = []

	for line in entry_list:
		ele=line.split()
		if model_ID>0:
			break
		if ele[0]=="ATOM":
			atom=(line[13:16].strip(' '))
			res=(line[17:20])
			chain_ID=line[21:26]
			chain=chain_ID[0]
			res_no=int(chain_ID[1:].strip(' '))
			chain_ID=(chain,res_no)
			new_pos=[float(line[30:37]),float(line[38:45]),float(line[46:53])]
			all_x.append(new_pos[0])
			all_y.append(new_pos[1])
			all_z.append(new_pos[2])
			all_pos.append(new_pos)
			all_lines.append(line)
			all_atom_type.append(atom[0])
			if chain_ID not in ID_dict.keys():
				ID_dict[chain_ID]=[PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index,value=1)]
			else:
				ID_dict[chain_ID].append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index,value=1))

			PDB_entries.append(PDB_atom(atom,res,chain_ID,new_pos[0],new_pos[1],new_pos[2],index=atom_index,value=1))
			atom_index+=1

		if ele[0]=="ENDMDL" and model_ID==0:
			model_ID+=1


	PROTEIN=[ID_dict,all_pos,all_lines, all_atom_type, PDB_entries, all_x, all_y, all_z]
				
	return PROTEIN

def load_dict(dict_name):
	
	if os.path.isfile(os.path.join('../data/DICT',dict_name)):
		with open(os.path.join('../data/DICT',dict_name)) as f:
			tmp_dict = json.load(f)
		res_count_dict={}
		for i in range (0,20):
			res_count_dict[i]=tmp_dict[str(i)]
	else:
		print ("dictionary not exist! initializing an empty one ..")
		res_count_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0}

	for key in res_count_dict:
		print (label_res_dict[(key)]+" "+str(res_count_dict[key]))

	return res_count_dict

def find_grid_points(all_x,all_y,all_z,grid_size=10):
	x_min=min(all_x)
	x_max=max(all_x)
	y_min=min(all_y)
	y_max=max(all_y)
	z_min=min(all_z)
	z_max=max(all_z)

	x_range=x_max-x_min
	y_range=y_max-y_min
	z_range=z_max-z_min

	num_of_grid_x=x_range/grid_size
	num_of_grid_y=y_range/grid_size
	num_of_grid_z=z_range/grid_size

	x_grids=[]
	y_grids=[]
	z_grids=[]

	x_c=0
	x_pos=x_min
	while(x_c<num_of_grid_x):
		x_grids.append(x_pos)
		x_pos=x_pos+grid_size
		x_c=x_c+1

	y_c=0
	y_pos=y_min
	while(y_c<num_of_grid_y):
		y_grids.append(y_pos)
		y_pos=y_pos+grid_size
		y_c=y_c+1

	z_c=0
	z_pos=z_min
	while(z_c<num_of_grid_z):
		z_grids.append(z_pos)
		z_pos=z_pos+grid_size
		z_c=z_c+1

	pos=[]

	for i in range(0,len(x_grids)):
		for j in range(0,len(y_grids)):
			for k in range(0,len(z_grids)):
				x=x_grids[i]
				y=y_grids[j]
				z=z_grids[k]
				pos.append([x,y,z])

	return pos


def pts_to_Xsmooth(PROTEIN,pts,atom_density,num_of_channels,pixel_size,box_size):
	num_3d_pixel = int(box_size/pixel_size)
	[ID_dict,all_pos,all_lines, all_atom_type, PDB_entries, all_x, all_y , all_z] = PROTEIN
	[pos,chain_ID,label]=pts
	backbone=ID_dict[chain_ID][0:4]
	deleted_res=ID_dict[chain_ID][4:]
	deleted_res_index = [atom.index for atom in deleted_res]
	res = backbone[0].res

	box=[]
	box_ori=[]
	X_smooth=[]
	reference=[]
	new_pos_in_box=[]
	atom_count=0
	valid_box = False
	box_x_min=-box_size/2
	box_x_max=+box_size/2
	box_y_min=-box_size/2
	box_y_max=+box_size/2
	box_z_min=-box_size/2
	box_z_max=+box_size/2

	get_position=get_position_dict(ID_dict[chain_ID])

	if set(get_position.keys())==label_atom_type_dict[label]:
		[reference,transform]=center_and_transform(label,get_position) 
		all_pos = numpy.array(all_pos)
		transformed_pos = ((all_pos - reference).dot(transform))-bias
		x_index = numpy.intersect1d(numpy.where(transformed_pos[:,0]>box_x_min),numpy.where(transformed_pos[:,0]<box_x_max))
		y_index = numpy.intersect1d(numpy.where(transformed_pos[:,1]>box_y_min),numpy.where(transformed_pos[:,1]<box_y_max))
		z_index = numpy.intersect1d(numpy.where(transformed_pos[:,2]>box_z_min),numpy.where(transformed_pos[:,2]<box_z_max))

		final_index = numpy.intersect1d(x_index,y_index)
		final_index = numpy.intersect1d(final_index,z_index)
		final_index = final_index.tolist()
		final_index = [ ind for ind in final_index if ind not in deleted_res_index]
		final_index = [ ind for ind in final_index if (all_atom_type[ind] =='C' or all_atom_type[ind]=='O' or all_atom_type[ind]=='S' or all_atom_type[ind]=='N') ]

		box_ori = [PDB_entries[i] for i in final_index]
		box_lines = [all_lines[i] for i in final_index] 
		new_pos_in_box = transformed_pos[final_index]
		atom_count = len(box_ori)
		threshold=(box_size**3)*atom_density

		if atom_count>threshold:
			valid_box = True

			# box_file = open('../data/BOX/'+d_name+'/'+PDB_ID+'_'+res+'_'+str(chain_ID[1])+'.pdb','w')
			# for l in box_lines:
			# 	box_file.write(l)
			
			sample=numpy.zeros((num_of_channels,num_3d_pixel,num_3d_pixel,num_3d_pixel))

			for i in range (0,len(box_ori)):
				atoms = box_ori[i]
				x=new_pos_in_box[i][0]
				y=new_pos_in_box[i][1]
				z=new_pos_in_box[i][2]

				x_new=x-box_x_min
				y_new=y-box_y_min
				z_new=z-box_z_min
				bin_x=int(numpy.floor(x_new/pixel_size))
				bin_y=int(numpy.floor(y_new/pixel_size))
				bin_z=int(numpy.floor(z_new/pixel_size))

				if(bin_x==num_3d_pixel):
					bin_x=num_3d_pixel-1
					
				if(bin_y==num_3d_pixel): 
					bin_y=num_3d_pixel-1
					
				if(bin_z==num_3d_pixel):
					bin_z=num_3d_pixel-1 
							   
				
				if atoms.atom[0]=='O':
					sample[0,bin_x,bin_y,bin_z] = sample[0,bin_x,bin_y,bin_z] + atoms.value
				elif atoms.atom[0]=='C':
					sample[1,bin_x,bin_y,bin_z] = sample[1,bin_x,bin_y,bin_z] + atoms.value
				elif atoms.atom[0]=='N':
					sample[2,bin_x,bin_y,bin_z] = sample[2,bin_x,bin_y,bin_z] + atoms.value
				elif atoms.atom[0]=='S':
					sample[3,bin_x,bin_y,bin_z] = sample[3,bin_x,bin_y,bin_z] + atoms.value
					
			X_smooth=numpy.zeros(sample.shape, dtype=theano.config.floatX)
			for j in range (0,4):
				X_smooth[j,:,:,:]=scipy.ndimage.filters.gaussian_filter(sample[j,:,:,:], sigma=0.6, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
				X_smooth[j,:,:,:]*=1000
			
	return X_smooth, label, reference, box_ori, new_pos_in_box, valid_box 

if __name__ == '__main__':

	d_name = sys.argv[1] # d_name can be 'train' or 'test'
	num_of_channels=4
	atom_density=0.01 # defalut = 0.01, desired threshold of atom density of boxes defined by num_of_atom / box volume
	box_size=20
	pixel_size=1

	PDB_DIR = '../data/PDB_family_'+d_name+'/'
	dat_dir = '../data/RAW_DATA/'+d_name+'/'
	dict_name = d_name+'_20AA_boxes.json'
	sample_block = 1000 if d_name=='train' else 100
	samples=[]

	if not os.path.exists('../data/DICT'):
		os.makedirs('../data/DICT')

	if not os.path.exists(dat_dir):
		os.makedirs(dat_dir)

	PDBs = load_pdb_set(d_name)
	res_count_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0}

	for PDB_ID in PDBs:    
		if os.path.isfile(PDB_DIR+PDB_ID+'.pdb'):
			print (PDB_ID)
			pdb_file = open(PDB_DIR+PDB_ID+'.pdb')
			infile = list(pdb_file)
			PROTEIN = grab_PDB(infile)
			[ID_dict, all_pos, all_lines, all_atom_type, PDB_entries, all_x, all_y , all_z] = PROTEIN

			my_kd_tree = scipy.spatial.KDTree(all_pos)
			pos = find_grid_points(all_x,all_y,all_z)

			ctr_pos=[]
			visited=set()
			actual_pos=[find_actual_pos(my_kd_tree, pos[i], PDB_entries) for i in range(len(pos))]

			for PDB_a in actual_pos:
				chain_ID=PDB_a.chain_ID
				res_atoms=ID_dict[chain_ID]
				res=PDB_a.res
				if res in res_label_dict.keys():
					label=res_label_dict[res]
					get_position=get_position_dict(res_atoms)
					if "CA" in get_position.keys():
						ctr=get_position["CA"]
						if ctr not in visited:
							visited.add(ctr)
							ctr_pos.append([ctr,chain_ID,label])

			for pts in ctr_pos:
				X_smooth, label, reference, box_ori, new_pos_in_box, valid_box  = pts_to_Xsmooth(PROTEIN,pts,atom_density,num_of_channels,pixel_size,box_size)
				if valid_box:
					res_container_dict[label].append(X_smooth)
					if(len(res_container_dict[label])==sample_block):
						sample_time_t = numpy.array(res_container_dict[label])
						res_container_dict[label]=[]
						sample_time_t.dump(dat_dir+'/'+label_res_dict[label]+"_"+str(res_count_dict[label])+'.dat')
						res_count_dict[label]+=1
						with open(os.path.join('../data/DICT',dict_name), 'w') as f:
							json.dump(res_count_dict, f)
							print ("dump dictionary...")
                 

			pdb_file.close()

	print ("done generating 20 amino acid boxes, storing dictionary..")
	with open(os.path.join('../data/DICT',dict_name), 'w') as f:
		json.dump(res_count_dict, f)

