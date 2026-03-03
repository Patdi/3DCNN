res_atom_type_dict={
'GLY':['N','CA','C','O'],
'CYS':['N','CA','C','O','CB','SG'],
'ARG':['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
'SER':['N','CA','C','O','CB','OG'],
'THR':['N','CA','C','O','CB','OG1','CG2'],
'LYS':['N','CA','C','O','CB','CG','CD','CE','NZ'],
'MET':['N','CA','C','O','CB','CG','SD','CE'],
'ALA':['N','CA','C','O','CB'],
'LEU':['N','CA','C','O','CB','CG','CD1','CD2'],
'ILE':['N','CA','C','O','CB','CG1','CG2','CD1'],
'VAL':['N','CA','C','O','CB','CG1','CG2'],
'ASP':['N','CA','C','O','CB','CG','OD1','OD2'],
'GLU':['N','CA','C','O','CB','CG','CD','OE1','OE2'],
'HIS':['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
'ASN':['N','CA','C','O','CB','CG','OD1','ND2'],
'PRO':['N','CA','C','O','CB','CG','CD'],
'GLN':['N','CA','C','O','CB','CG','CD','OE1','NE2'],
'PHE':['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
'TRP':['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
'TYR':['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH']
}

label_atom_type_dict={
18:set(['N','CA','C','O']),
19:set(['N','CA','C','O','CB','SG']),
2:set(['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2']),
5:set(['N','CA','C','O','CB','OG']),
6:set(['N','CA','C','O','CB','OG1','CG2']),
1:set(['N','CA','C','O','CB','CG','CD','CE','NZ']),
13:set(['N','CA','C','O','CB','CG','SD','CE']),
9:set(['N','CA','C','O','CB']),
11:set(['N','CA','C','O','CB','CG','CD1','CD2']),
12:set(['N','CA','C','O','CB','CG1','CG2','CD1']),
10:set(['N','CA','C','O','CB','CG1','CG2']),
3:set(['N','CA','C','O','CB','CG','OD1','OD2']),
4:set(['N','CA','C','O','CB','CG','CD','OE1','OE2']),
0:set(['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2']),
7:set(['N','CA','C','O','CB','CG','OD1','ND2']),
17:set(['N','CA','C','O','CB','CG','CD']),
8:set(['N','CA','C','O','CB','CG','CD','OE1','NE2']),
14:set(['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ']),
16:set(['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2']),
15:set(['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH']),
}

res_group_dict={
    0:'group1',1:'group1',2:'group1',
    3:'group2',4:'group2',
    5:'group3',6:'group3',7:'group3',8:'group3',
    9:'group4',10:'group4',11:'group4',12:'group4',13:'group4',
    14:'group5',15:'group5',16:'group5',
    17:'group6',18:'group6',
    19:'group7'
    }

peptide_bond_energy=float(615+305)/2

AA_bond_dict={
'GLY':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
},

'CYS':{ 
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','SG'):259,
},

'ARG':{ 
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD'):347,
('CD','NE'):305,
('NE','CZ'):305,
('CZ','NH1'):305,
('CZ','NH2'):615,
},

'SER':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','OG'):358,
},

'THR':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','OG1'):358,
('CB','CG2'):347,
},

'LYS':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD'):347,
('CD','CE'):347,
('CE','NZ'):305,
},

'MET':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','SD'):259,
('SD','CE'):259,
},

'ALA':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
},

'LEU':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD1'):347,
('CG','CD2'):347
},

'ILE':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG1'):347,
('CB','CG2'):347,
('CG1','CD1'):347,
},

'VAL':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG1'):347,
('CB','CG2'):347,
},

'ASP':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','OD1'):745,
('CG','OD2'):358
},

'GLU':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD'):347,
('CD','OE1'):745,
('CD','OE2'):358
},

'HIS':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','ND1'):305,
('ND1','CE1'):615,
('CE1','NE2'):305,
('NE2','CD2'):305,
('CD2','CG'):614,
},

'ASN':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','OD1'):745,
('CG','ND2'):305
},

'PRO':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD'):347,
('CD','N'):305
},

'GLN':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD'):347,
('CD','OE1'):745,
('CD','NE2'):305
},

'PHE':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD1'):614,
('CD1','CE1'):347,
('CE1','CZ'):614,
('CZ','CE2'):347,
('CE2','CD2'):614,
('CD2','CG'):347,
},

'TRP':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD1'):614,
('CD1','NE1'):305,
('NE1','CE2'):305,
('CE2','CD2'):614,
('CD2','CG'):305,
('CE2','CZ2'):347,
('CZ2','CH2'):614,
('CH2','CZ3'):347,
('CZ3','CE3'):614,
('CE3','CD2'):347,
},

'TYR':{
('N','CA'):305,
('CA','C'):347,
('C','O'):745,
('CA','CB'):347,
('CB','CG'):347,
('CG','CD1'):614,
('CD1','CE1'):347,
('CE1','CZ'):614,
('CZ','OH'):358,
('CZ','CE2'):347,
('CE2','CD2'):614,
('CD2','CG'):347,
}

}

abrev={'HIS':'H','LYS':'K','ARG':'R','ASP':'D','GLU':'E','SER':'S','THR':'T','ASN':'N','GLN':'Q','ALA':'A','VAL':'V','LEU':'L','ILE':'I','MET':'M','PHE':'F','TYR':'Y','TRP': 'W','PRO': 'P','GLY': 'G', 'CYS': 'C'}


label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}

letter1_3_dict={'H':'HIS','K':'LYS','R':'ARG','D':'ASP','E':'GLU','S':'SER','T':'THR','N':'ASN','Q':'GLN','A':'ALA','V':'VAL','L':'LEU','I':'ILE','M':'MET','F':'PHE','Y':'TYR','W':'TRP','P':'PRO','G':'GLY','C':'CYS'}
    
bias=[-0.558,-0.73,1.226]