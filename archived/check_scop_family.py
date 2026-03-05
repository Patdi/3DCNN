import collections

pdb_all_file = open("../data/SCOP/dir.cla.scope.2.07-stable.txt")
test_pdb = open('../data/PDB_test.txt')
train_pdb = open('../data/PDB_train.txt')

scop_dict=collections.defaultdict(set)
pdb_dict=collections.defaultdict(set)
test_fam=set()
train_fam=set()
train_id = set()
test_id = set()

pdb_all_list = list(pdb_all_file)

for line in pdb_all_list[5:]:
	ele = line.split()
	sid = ele[0]
	pdb = ele[1]
	fam = ele[3]
	scop_dict[fam].add(pdb)
	pdb_dict[pdb].add(fam)

for line in train_pdb:
	pdb = line.strip('\n')
	train_fam=train_fam.union(pdb_dict[pdb])
	train_id.add(pdb)


for line in test_pdb:
	pdb = line.strip('\n')
	test_fam=test_fam.union(pdb_dict[pdb])
	test_id.add(pdb)

# ensure no test chain is in the same family as any training chain
inter=train_fam.intersection(test_fam)
# should be empty!
print "intersection of training and test families:"
print inter


