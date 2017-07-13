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
	labels = np.loadtxt(fname,dtype=str,skiprows=10,usecols=(0,1,2,3,6,7,8))
	labels = map(lambda x: 
					[reduce(lambda a,b: a+' '+b, 
						filter(lambda y: y != '-', x[:4]))] + list(x[4:]), labels)
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
	for l in lines:
		if l[1] not in phone2state:
			l[1] = l[1].split()[0]
	states = map(lambda x: phone2state[x[1]][int(x[0])], lines)
	return (list(states))

def genDataset(DB_path, filelist, feat_path, stseg_path, mdef_fname, context_len=None):
	global done
	files = np.loadtxt(DB_path+filelist,dtype=str)
	files = map(lambda x: DB_path+feat_path+x+'.mls',files)
	
	train_files = filter(lambda x: 'tr' in x.split('/')[-3] and
									'wv1' == x.split('.')[-3], files)
	test_files = filter(lambda x: 'si_et_20' in x.split('/')[-3]  and
									'wv1' == x.split('.')[-3], files)
	dev_files = filter(lambda x: 'si_dt_20' in x.split('/')[-3] and
									'wv1' == x.split('.')[-3], files)

	stseg_files_train = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',train_files)
	stseg_files_test = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',test_files)
	stseg_files_dev = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',dev_files)
	stseg_files_train = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_train)
	stseg_files_test = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_test)
	stseg_files_dev = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_dev)

	stseg_files = stseg_files_train + stseg_files_dev + stseg_files_test
	print "Training Files: %d 	Dev Files: %d	Testing Files: %d" % (len(stseg_files_train), len(stseg_files_dev), len(stseg_files_test))
	
	phone2state = read_sen_labels_from_mdef(mdef_fname,onlyPhone=False)

	X_Train = []
	Y_Train = []
	X_Test = []
	Y_Test = []
	X_Dev = []
	Y_Dev = []
	framePos_Train = []
	framePos_Test = []
	framePos_Dev = []
	filenames_Train = []
	filenames_Test = []
	filenames_Dev = []
	# allData = []
	# allLabels = []
	pos = 0
	scaler = StandardScaler(copy=False,with_std=False)
	n_states = np.max(phone2state.values())+1
	print n_states
	state_freq_Train = [0]*n_states
	state_freq_Dev = [0]*n_states
	state_freq_Test = [0]*n_states
	for i in range(len(stseg_files)):
		sys.stdout.write("\r%d/%d 	" % (i,len(stseg_files)))
		sys.stdout.flush()
		f = stseg_files[i]
		
		data_file = filter(lambda x: f[:-9] in x, files)[0]
		
		data = np.loadtxt(data_file).astype('float32')
		labels = frame2state(DB_path + stseg_path + f, phone2state,onlyPhone=False)

		nFrames = min(len(labels), data.shape[0])
		data = data[:nFrames]
		data = scaler.fit_transform(data)
		labels = labels[:nFrames]
		if context_len != None:
			pad_top = np.zeros((context_len,data.shape[1]))
			pad_bot = np.zeros((context_len,data.shape[1]))
			padded_data = np.concatenate((pad_top,data),axis=0)
			padded_data = np.concatenate((padded_data,pad_bot),axis=0)

			data = []
			for j in range(context_len,len(padded_data) - context_len):
				new_row = padded_data[j - context_len: j + context_len + 1]
				new_row = new_row.flatten()
				data.append(new_row)

		if i < len(stseg_files_train):
			# print '\n train'
			frames = framePos_Train
			allData = X_Train
			allLabels = Y_Train
			filenames = filenames_Train
			state_freq = state_freq_Train
		elif i < len(stseg_files_train) + len(stseg_files_dev):
			# print '\n dev'
			frames = framePos_Dev
			allData = X_Dev
			allLabels = Y_Dev
			filenames = filenames_Dev
			state_freq = state_freq_Dev
		else:
			# print '\n test'
			frames = framePos_Test
			allData = X_Test
			allLabels = Y_Test
			filenames = filenames_Test
			state_freq = state_freq_Test
		for l in labels:
			state_freq[l] += 1
		filenames.append(data_file)
		frames.append(nFrames)
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
		np.save('wsj0_phonelabels_bracketed_train.npy',X_Train)
		np.save('wsj0_phonelabels_bracketed_test.npy',X_Test)
		np.save('wsj0_phonelabels_bracketed_dev.npy',X_Dev)
		np.save('wsj0_phonelabels_bracketed_train_labels.npy',Y_Train)
		np.save('wsj0_phonelabels_bracketed_test_labels.npy',Y_Test)
		np.save('wsj0_phonelabels_bracketed_dev_labels.npy',Y_Dev)
		np.savez('wsj0_phonelabels_bracketed_meta.npz',framePos_Train=framePos_Train,
														framePos_Test=framePos_Test,
														framePos_Dev=framePos_Dev,
														filenames_Train=filenames_Train,
														filenames_Dev=filenames_Dev,
														filenames_Test=filenames_Test,
														state_freq_Train=state_freq_Train,
														state_freq_Dev=state_freq_Dev,
														state_freq_Test=state_freq_Test)
	else:	
		np.save('wsj0_phonelabels_train.npy',X_Train)
		np.save('wsj0_phonelabels_test.npy',X_Test)
		np.save('wsj0_phonelabels_dev.npy',X_Dev)
		np.save('wsj0_phonelabels_train_labels.npy',Y_Train)
		np.save('wsj0_phonelabels_test_labels.npy',Y_Test)
		np.save('wsj0_phonelabels_dev_labels.npy',Y_Dev)
		np.savez('wsj0_phonelabels_meta.npz',framePos_Train=framePos_Train,
														framePos_Test=framePos_Test,
														framePos_Dev=framePos_Dev,
														filenames_Train=filenames_Train,
														filenames_Dev=filenames_Dev,
														filenames_Test=filenames_Test,
														state_freq_Train=state_freq_Train,
														state_freq_Dev=state_freq_Dev,
														state_freq_Test=state_freq_Test)
	# done = 1

def normalizeByUtterance():
	data = np.load('wsj0_phonelabels_bracketed_data.npy')
	nFrames = np.load('wsj0_phonelabels_bracketed_meta.npz')['framePos_Train']
	# print 'calculating frame indices...'
	# nFrames = map(lambda i: sum(nFrames[:i]),xrange(len(nFrames)))
	print 'normalizing...'
	scaler = StandardScaler(copy=False)
	print data
	pos = 0
	for i in xrange(len(nFrames)):
		sys.stdout.write("\rnormalizing utterance no %d " % i)
		sys.stdout.flush()
		data[pos:nFrames[i]] = scaler.fit_transform(data[pos:nFrames[i]])
		pos = nFrames[i]
	print data
#print(read_sen_labels_from_mdef('../wsj_all_cd30.mllt_cd_cont_4000/mdef'))
# frame2state('../wsj/wsj0/statesegdir/40po031e.wv2.flac.stseg.txt', '../wsj_all_cd30.mllt_cd_cont_4000/mdef')
genDataset('../wsj/wsj0/','etc/wsj0_train.fileids','feat_ci_mls/','stateseg_ci_dir/','../en_us.ci_cont/mdef')
# normalizeByUtterance()
# ../wsj/wsj0/feat_mls/11_6_1/wsj0/sd_dt_20/00b/00bo0t0e.wv1.flac.mls 00bo0t0e.wv1.flac.stseg.txt