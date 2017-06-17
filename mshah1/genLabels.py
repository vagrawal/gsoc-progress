import numpy as np
from functools import reduce
import os
import threading
import time
import sys
from sklearn.preprocessing import StandardScaler
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
	
	speakers = list(set(map(lambda x: x.split('/')[-2],files)))
	lenTrain = len(speakers) * 3 / 4
	train_speakers = speakers[:lenTrain]
	train_files = filter(lambda x: x.split('/')[-2] in train_speakers, files)
	test_files = filter(lambda x: x not in train_files, files)

	print "Training Speakers: %d 	Testing Speakers: %d" % (lenTrain, len(speakers)-lenTrain)
	

	stseg_files_train = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',train_files)
	stseg_files_test = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',test_files)
	stseg_files_train = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_train)
	stseg_files_test = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_test)
	stseg_files = stseg_files_train + stseg_files_test
	print "Training Files: %d 	Testing Files: %d" % (len(stseg_files_train), len(stseg_files_test))
	phone2state = read_sen_labels_from_mdef(mdef_fname)
	# X_Train = []
	Y_Train = []
	# X_Test = []
	Y_Test = []
	framePos_Train = []
	framePos_Test = []
	allData = []
	# allLabels = []
	pos = 0
	for i in range(len(stseg_files)):
		sys.stdout.write("\r%d/%d 	" % (i,len(stseg_files)))
		sys.stdout.flush()
		f = stseg_files[i]
		data_file = filter(lambda x: f[:-9] in x, files)[0]
		
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

		if i < len(stseg_files_train):
			allLabels = Y_Train
			# allData = X_Train
			frames = framePos_Train
		else:
			allLabels = Y_Test
			# allData = X_Test
			frames = framePos_Test
		frames.append(pos + nFrames)
		allData += list(data)
		allLabels += list(labels)
		pos += nFrames
		assert(len(allLabels) == len(allData))
	# print allData
	print len(allData), len(allLabels)

	
	# np.savez('wsj0_phonelabels_NFrames',NFrames_Train=NFrames_Train,NFrames_Test=NFrames_Test)
	# t = threading.Thread(target=ping)
	# t.start()
	if context_len != None:
		np.save('wsj0_phonelabels_bracketed_data.npy',allData)
		np.savez('wsj0_phonelabels_bracketed_meta.npz',Y_Train=Y_Train,Y_Test=Y_Test,framePos_Train=framePos_Train,framePos_Test=framePos_Test)
	else:	
		np.savez('wsj0_phonelabels',X_Train=X_Train,Y_Train=Y_Train,X_Test=X_Test,Y_Test=Y_Test,framePos_Train=framePos_Train,framePos_Test=framePos_Test)
	# done = 1

def normalizeByUtterance():
	data = np.load('wsj0_phonelabels.npz')
	nFrames = data['NFrames_Train']
	data = data['X_Train']
	# print 'calculating frame indices...'
	# nFrames = map(lambda i: sum(nFrames[:i]),xrange(len(nFrames)))
	print 'normalizing...'
	scaler = StandardScaler(copy=False)
	print data
	pos = 0
	for i in xrange(len(nFrames)):
		sys.stdout.write("\rnormalizing utterance no %d " % i)
		sys.stdout.flush()
		data[pos:pos+nFrames[i]] = scaler.fit_transform(data[pos:pos+nFrames[i]])
		pos += nFrames[i]
	print data
#print(read_sen_labels_from_mdef('../wsj_all_cd30.mllt_cd_cont_4000/mdef'))
# frame2state('../wsj/wsj0/statesegdir/40po031e.wv2.flac.stseg.txt', '../wsj_all_cd30.mllt_cd_cont_4000/mdef')
genDataset('../wsj/wsj0/','wsj0.mlslist','statesegdir/','../wsj_all_cd30.mllt_cd_cont_4000/mdef',context_len=4)
# normalizeByUtterance()
# ../wsj/wsj0/feat_mls/11_6_1/wsj0/sd_dt_20/00b/00bo0t0e.wv1.flac.mls 00bo0t0e.wv1.flac.stseg.txt