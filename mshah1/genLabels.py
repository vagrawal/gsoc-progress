import numpy as np
from functools import reduce
import os
import threading
import time
import sys
from sklearn.preprocessing import StandardScaler
import utils
from keras.preprocessing.sequence import pad_sequences
import ctc
import fst
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

def loadDict(filename):
	def mySplit(line):
		line = line.split()
		for i in range(1,len(line)):
			line[i] = "{0}1 {0}2 {0}3".format(line[i])
		line = [line[0], reduce(lambda x,y: x+ ' ' +y, line[1:])]
		return line
	with open(filename) as f:
		d = f.readlines()
	d = map(lambda x: x.split(),d)
	myDict = {}
	for r in d:
		myDict[r[0]] = r[1:]
	return myDict

def loadTrans(trans_file,pDict):
	trans = {}
	with open(trans_file) as f:
		lines = f.readlines()
	for line in lines:
		line = line.split()
		fname = line[-1][1:-1]
		labels = map(lambda x: pDict.setdefault(x,-1), line[:-1])
		labels = filter(lambda x: x!=-1, labels)
		if labels == []:
			continue
		labels = reduce(lambda x,y: x + y, labels)
		trans[fname] = labels
	return trans

def trans2labels(trans,phone2state):
	d = {}
	for u in trans:
		labels = trans[u]
		labels = map(lambda x: phone2state[x], labels)
		labels = reduce(lambda x,y: x + y, labels)
		d[u] = labels
	return d

def genDataset(DB_path, train_flist, dev_flist, test_flist, 
				feat_path, stseg_path, mdef_fname, context_len=None, 
				keep_utts=False, ctc_labels=False, pDict_file=None,
				trans_file=None, make_graph=False):
	global done
	train_files = np.loadtxt(DB_path+train_flist,dtype=str)
	train_files = map(lambda x: DB_path+feat_path+x+'.mls',train_files)
	train_files = filter(lambda x: 'tr_' in x and
									'wv1' in x, train_files)
	test_files = np.loadtxt(DB_path+test_flist,dtype=str)
	test_files = map(lambda x: DB_path+feat_path+x+'.mls',test_files)
	dev_files = np.loadtxt(DB_path+dev_flist,dtype=str)
	dev_files = map(lambda x: DB_path+feat_path+x+'.mls',dev_files)

	phone2state = read_sen_labels_from_mdef(mdef_fname,onlyPhone=False)

	if ctc_labels:
		pDict = loadDict(pDict_file)
		trans = loadTrans(trans_file,pDict)
		label_dict = trans2labels(trans, phone2state)
		stseg_files_train = filter(lambda x: x.split('/')[-1][:-4] in label_dict, train_files)[:2400]
		stseg_files_test = filter(lambda x: x.split('/')[-1][:-4] in label_dict, test_files)[:110]
		stseg_files_dev = filter(lambda x: x.split('/')[-1][:-4] in label_dict, dev_files)[:33]
		stseg_files = stseg_files_train + stseg_files_dev + stseg_files_test
		print "Training Files: %d 	Dev Files: %d	Testing Files: %d" % (len(stseg_files_train), len(stseg_files_dev), len(stseg_files_test))
	else:
		stseg_files_train = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',train_files)
		stseg_files_test = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',test_files)
		stseg_files_dev = map(lambda x: x.split('/')[-1][:-3]+'stseg.txt',dev_files)
		stseg_files_train = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_train)[:2400]
		stseg_files_test = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_test)[:110]
		stseg_files_dev = filter(lambda x: os.path.exists(DB_path + stseg_path + x), stseg_files_dev)[:33]

		stseg_files = stseg_files_train + stseg_files_dev + stseg_files_test
		print "Training Files: %d 	Dev Files: %d	Testing Files: %d" % (len(stseg_files_train), len(stseg_files_dev), len(stseg_files_test))

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
	active_states_Train = []
	active_states_Test = []
	active_states_Dev = []
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
		if i < len(stseg_files_train):
			# print '\n train'
			frames = framePos_Train
			allData = X_Train
			allLabels = Y_Train
			filenames = filenames_Train
			state_freq = state_freq_Train
			files = train_files
			active_state = active_states_Train
		elif i < len(stseg_files_train) + len(stseg_files_dev):
			# print '\n dev'
			frames = framePos_Dev
			allData = X_Dev
			allLabels = Y_Dev
			filenames = filenames_Dev
			state_freq = state_freq_Dev
			files = dev_files
			active_state = active_states_Dev
		else:
			# print '\n test'
			frames = framePos_Test
			allData = X_Test
			allLabels = Y_Test
			filenames = filenames_Test
			state_freq = state_freq_Test
			files = test_files
			active_state = active_states_Test

		sys.stdout.write("\r%d/%d 	" % (i,len(stseg_files)))
		sys.stdout.flush()
		f = stseg_files[i]
		
		[data_file] = filter(lambda x: f[:-9] in x, files)
		data = utils.readMFC(data_file,40).astype('float32')
		data = scaler.fit_transform(data)

		if ctc_labels:
			labels = label_dict[data_file.split('/')[-1][:-4]]
			if make_graph:
				t = ctc.genBigGraph(ctc.ran_lab_prob(data.shape[0]),
										set(labels),
										data.shape[0])
				t2 = ctc.gen_utt_graph(trans[data_file.split('/')[-1][:-4]],phone2state)
				assert set([e for (e,_) in t.osyms.items()]) == set([e for (e,_) in t2.isyms.items()])
				t.osyms = t2.isyms
				# ctc.print_graph(t)
				# print [e for (e,_) in t.osyms.items()]
				# print [e for (e,_) in t2.isyms.items()]
				t3 = t >> t2
				parents = ctc.gen_parents_dict(t3)
				y_t_s = ctc.make_prob_dict(t3,data.shape[0],n_states)
				# print y_t_s
				# alpha = ctc.calc_alpha(data.shape[0],labels,y_t_s)
				# beta = ctc.calc_beta(data.shape[0],labels,y_t_s)
				active_state.append(y_t_s)
			# print active_state
			nFrames = data.shape[0]
		else:
			labels = frame2state(DB_path + stseg_path + f, phone2state,onlyPhone=False)
			nFrames = min(len(labels), data.shape[0])
			data = data[:nFrames]
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

		for l in labels:
			state_freq[l] += 1
		filenames.append(data_file)
		frames.append(nFrames)
		if keep_utts:
			allData.append(data)
			allLabels.append(np.array(labels) + 1)			
		else:
			allData += list(data)
			allLabels += list(labels)
		pos += nFrames
		if not ctc_labels:
			assert(len(allLabels) == len(allData))
	# print allData
	print len(allData), len(allLabels)
	if keep_utts:
		X_Train = pad_sequences(X_Train,maxlen=1000,dtype='float32',padding='post')
		Y_Train = pad_sequences(Y_Train,maxlen=1000,dtype='float32',padding='post',value=n_states)
		Y_Train = Y_Train.reshape(Y_Train.shape[0],Y_Train.shape[1],1)
		X_Dev = pad_sequences(X_Dev,maxlen=1000,dtype='float32',padding='post')
		Y_Dev = pad_sequences(Y_Dev,maxlen=1000,dtype='float32',padding='post',value=n_states)
		Y_Dev = Y_Dev.reshape(Y_Dev.shape[0],Y_Dev.shape[1],1)
		X_Test = pad_sequences(X_Test,maxlen=1000,dtype='float32',padding='post')
		Y_Test = pad_sequences(Y_Test,maxlen=1000,dtype='float32',padding='post',value=n_states)
		Y_Test = Y_Test.reshape(Y_Test.shape[0],Y_Test.shape[1],1)
	# np.savez('wsj0_phonelabels_NFrames',NFrames_Train=NFrames_Train,NFrames_Test=NFrames_Test)
	# t = threading.Thread(target=ping)
	# t.start()
	if context_len != None:
		# np.save('wsj0_phonelabels_bracketed_mini_train.npy',X_Train)
		# np.save('wsj0_phonelabels_bracketed_mini_test.npy',X_Test)
		# np.save('wsj0_phonelabels_bracketed_mini_dev.npy',X_Dev)
		np.save('wsj0_phonelabels_bracketed_mini_trans_train_labels.npy',Y_Train)
		np.save('wsj0_phonelabels_bracketed_mini_trans_test_labels.npy',Y_Test)
		np.save('wsj0_phonelabels_bracketed_mini_trans_dev_labels.npy',Y_Dev)
		if make_graph:
			np.save('wsj0_phonelabels_bracketed_train_active.npy',active_states_Train)
			np.save('wsj0_phonelabels_bracketed_test_active.npy',active_states_Test)
			np.save('wsj0_phonelabels_bracketed_dev_active.npy',active_states_Dev)
		# np.savez('wsj0_phonelabels_bracketed_mini_meta.npz',framePos_Train=framePos_Train,
		# 												framePos_Test=framePos_Test,
		# 												framePos_Dev=framePos_Dev,
		# 												filenames_Train=filenames_Train,
		# 												filenames_Dev=filenames_Dev,
		# 												filenames_Test=filenames_Test,
		# 												state_freq_Train=state_freq_Train,
		# 												state_freq_Dev=state_freq_Dev,
		# 												state_freq_Test=state_freq_Test)
	else:	
		np.save('wsj0_phonelabels_train.npy',X_Train)
		np.save('wsj0_phonelabels_test.npy',X_Test)
		np.save('wsj0_phonelabels_dev.npy',X_Dev)
		np.save('wsj0_phonelabels_train_labels.npy',Y_Train)
		np.save('wsj0_phonelabels_train_active.npy',active_states_Train)
		np.save('wsj0_phonelabels_test_labels.npy',Y_Test)
		np.save('wsj0_phonelabels_test_active.npy',active_states_Test)
		np.save('wsj0_phonelabels_dev_labels.npy',Y_Dev)
		np.save('wsj0_phonelabels_dev_active.npy',active_states_Dev)
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
genDataset('../wsj/wsj0/','etc/wsj0_train.fileids','etc/wsj0_dev.fileids','etc/wsj0_test.fileids','feat_cd_mls/','stateseg_ci_dir/','../en_us.ci_cont/mdef',
			keep_utts=True, ctc_labels=True, context_len=5, 
			trans_file='../wsj/wsj0/etc/wsj0.transcription', 
			pDict_file='../wsj/wsj0/etc/cmudict.0.6d.wsj0')
# normalizeByUtterance()
# ../wsj/wsj0/feat_mls/11_6_1/wsj0/sd_dt_20/00b/00bo0t0e.wv1.flac.mls 00bo0t0e.wv1.flac.stseg.txt