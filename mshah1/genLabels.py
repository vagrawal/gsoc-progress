import numpy as np
from functools import reduce
import os
import threading
import time
import sys
done = 0
def ping():
	curr_time = int(time.time())
	while done == 0:
		if (int(time.time()) != curr_time):
			curr_time = int(time.time())
			sys.stdout.write('.')
			sys.stdout.flush()


def read_sen_labels_from_mdef(fname, onlyPhone=True):
	labels = np.loadtxt(fname,dtype=str,skiprows=10,usecols=(0,6,7,8))
	if onlyPhone:
		labels = labels[:44]
	phone2state = {}
	for r in labels:
		phone2state[r[0]] = map(int, r[1:])
	return phone2state

def frame2state(fname, phone2state, onlyPhone=True):
	with open(fname,'r') as f:
		lines = f.readlines()[2:]
	lines = map(lambda x: x.split()[2:], lines)
	if onlyPhone:
		lines = map(lambda x: x[:2],lines)
	else:
		lines = map(lambda x: [x[0], reduce(lambda a,b: a+' '+b,x[1:])],lines)
	states = map(lambda x: phone2state[x[1]][int(x[0])], lines)
	return (list(states))

def genDataset(DB_path, filelist, stseg_path, mdef_fname, context_len=None):
	global done
	files = np.loadtxt(DB_path+filelist,dtype=str)
	files = map(lambda x: DB_path+x,files)
	fileWOpath = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',files)
	stseg_files = filter(lambda x: os.path.exists(DB_path + stseg_path + x), fileWOpath)
	phone2state = read_sen_labels_from_mdef(mdef_fname)
	allData = []
	allLabels = []
	for i in range(len(stseg_files)):
		f = stseg_files[i]
		data_file = filter(lambda x: f[:-9] in x, files)[0]
		print "%d/%d" % (i,len(stseg_files)), data_file, f
		data = np.loadtxt(data_file)
		labels = frame2state(DB_path + stseg_path + f, phone2state)
		nFrames = min(len(labels), data.shape[0])
		data = data[:nFrames]
		labels = labels[:nFrames]
		if context_len != None:
			pad_top = np.zeros((context_len,data.shape[1])) + data[0]
			pad_bot = np.zeros((context_len,data.shape[1])) + data[-1]
			padded_data = np.concatenate((pad_top,data),axis=0)
			padded_data = np.concatenate((padded_data,pad_bot),axis=0)

			data = []
			for i in range(context_len,len(padded_data) - context_len):
				new_row = padded_data[i - context_len: i + context_len + 1]
				new_row = new_row.flatten()
				data.append(new_row)
		allData += list(data)
		allLabels += list(labels)
		assert(len(allLabels) == len(allData))
	# print allData
	print len(allData), len(allLabels)
	lenTrain = len(allData) * 3 / 4
	X_Train = allData[:lenTrain]
	Y_Train = allLabels[:lenTrain]
	X_Test = allData[lenTrain:]
	Y_Test = allLabels[lenTrain:]
	# t = threading.Thread(target=ping)
	# t.start()
	if context_len != None:
		np.savez('wsj0_phonelabels_bracketed',X_Train=X_Train,Y_Train=Y_Train,X_Test=X_Test,Y_Test=Y_Test)
	else:	
		np.savez('wsj0_phonelabels',X_Train=X_Train,Y_Train=Y_Train,X_Test=X_Test,Y_Test=Y_Test)
	done = 1
#print(read_sen_labels_from_mdef('../wsj_all_cd30.mllt_cd_cont_4000/mdef'))
# frame2state('../wsj/wsj0/statesegdir/40po031e.wv2.flac.stseg.txt', '../wsj_all_cd30.mllt_cd_cont_4000/mdef')
genDataset('../wsj/wsj0/','wsj0.mlslist','statesegdir/','../wsj_all_cd30.mllt_cd_cont_4000/mdef', context_len=4)
# ../wsj/wsj0/feat_mls/11_6_1/wsj0/sd_dt_20/00b/00bo0t0e.wv1.flac.mls 00bo0t0e.wv1.flac.stseg.txt