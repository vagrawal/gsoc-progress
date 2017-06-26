import numpy as np
def read_sen_labels_from_mdef(fname, onlyPhone=True):
	phone2state = {}
	if onlyPhone:
		labels = np.loadtxt(fname,dtype=str,skiprows=10,usecols=(0,6,7,8))
		labels = labels[:44]
	
		for r in labels:
			phone2state[r[0]] = map(int, r[1:])
	else:
		labels = np.loadtxt(fname,dtype=str,skiprows=10,usecols=(0,1,2,3,6,7,8))
		for r in labels:
			phone2state[reduce(lambda x,y: x+' '+y,
							filter(lambda a: a!='-', r[:4]))] = map(int, r[4:])
	return phone2state
def changeMapping(mdef):
	phone2state = read_sen_labels_from_mdef(mdef)
	for p in phone2state:
		phone2state[p] = phone2state[p.split()[0]]
	# print phone2state
	with open(mdef) as f:
		lines = f.readlines()
	with open(mdef+'_mapped','w') as f:
		for i in range(54):
			f.write(lines[i])
		for i in range(54,len(lines)):
			line = lines[i].split()
			line[6:9] = phone2state[line[0]]
			f.write('   %s  %s  %s %s    %s    %s    %d    %d    %d %s\n' % tuple(line))
changeMapping('mdef')