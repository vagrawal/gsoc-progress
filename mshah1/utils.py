import numpy as np
import struct
import matplotlib.pyplot as plt
import pylab as pl
from sys import stdout
import os
from keras.preprocessing.sequence import pad_sequences
def readMFC(fname,nFeats):
	data = []
	with open(fname,'rb') as f:
		v = f.read(4)
		head = struct.unpack('I',v)[0]
		v = f.read(nFeats * 4)
		while v:
			frame = list(struct.unpack('%sf' % nFeats, v))
			data .append(frame)
			v = f.read(nFeats * 4)
	data = np.array(data)
	# print data.shape, head
	assert(data.shape[0] * data.shape[1] == head)
	return data

def _gen_bracketed_data(x,y,nFrames,
						context_len,fix_length,
						for_CTC):
	max_len = ((np.max(nFrames) + 50)/100) * 100 #rounding off to the nearest 100
	batch_size = None
	while 1:
		pos = 0
		nClasses = np.max(y) + 1
		if for_CTC:
			alldata = []
			alllabels = []
		for i in xrange(len(nFrames)):
			data = x[pos:pos + nFrames[i]]
			labels = y[pos:pos + nFrames[i]]

			# if len(labels.shape) == 1:
			# 	labels = to_categorical(labels,num_classes=nClasses)
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
				data = np.array(data)
			if for_CTC:
				if batch_size != None:
					alldata.append(data)
					alllabels.append(labels)
				
					if len(alldata) == batch_size:
						alldata = np.array(alldata)
						alllabels = np.array(alllabels)
						if fix_length:
							alldata = pad_sequences(alldata,maxlen=1000,dtype='float32')
							alllabels = pad_sequences(alllabels,maxlen=1000,dtype='float32',value=0)
						inputs = {'x': alldata,
								'y': alllabels,
								'x_len': np.array(map(lambda x: len(x), alldata)),
								'y_len': np.array(map(lambda x: len(x), alllabels))}
						outputs = {'ctc': np.zeros([batch_size])}
						yield (inputs,outputs)
						alldata = []
						alllabels = []
				else:
					data = np.array([data])
					labels = np.array([labels])
					inputs = {'x': data,
								'y': labels,
								'x_len': [data.shape[0]],
								'y_len': [labels.shape[0]]}
					outputs = {'ctc': np.zeros([batch_size])}
					yield (inputs,outputs)
			else:
				yield (data,labels)
			pos += nFrames[i]

def gen_bracketed_data(context_len=None,fix_length=False,
						for_CTC=False):
	return lambda x,y,nf: _gen_bracketed_data(x,y,nf,context_len,fix_length,
						for_CTC)

def plotFromCSV(modelName):
	data = np.loadtxt(modelName+'.csv',skiprows=1,delimiter=',')
	epoch = data[:,[0]]
	acc = data[:,[1]]
	loss = data[:,[2]]
	val_acc = data[:,[4]]
	val_loss = data[:,[5]]

	fig, ax1 = plt.subplots()
	ax1.plot(acc)
	ax1.plot(val_acc)
	ax2 = ax1.twinx()
	ax2.plot(loss,color='r')
	ax2.plot(val_loss,color='g')
	plt.title('model loss & accuracy')
	ax1.set_ylabel('accuracy')
	ax2.set_ylabel('loss')
	ax1.set_xlabel('epoch')
	ax1.legend(['training acc', 'testing acc'])
	ax2.legend(['training loss', 'testing loss'])
	fig.tight_layout()
	plt.savefig(modelName+'.png')
	plt.clf()

def writeSenScores(filename,scores,freqs,weight,offset):
	n_active = scores.shape[1]
	s = ''
	s = """s3
version 0.1
mdef_file ../../en_us.cd_cont_4000/mdef
n_sen 4138
logbase 1.000100
endhdr
"""
	s += struct.pack('I',0x11223344)
	# print freqs
	# scores /= freqs + (1.0 / len(freqs))
	scores = np.log(scores)/np.log(1.0001)
	scores *= -1
	scores -= np.min(scores,axis=1).reshape(-1,1)
	# scores = scores.astype(int)
	scores *= weight
	scores += offset
	truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
	vf = np.vectorize(truncateToShort)
	scores = vf(scores)
	# scores /= np.sum(scores,axis=0)
	for r in scores:
		# print np.argmin(r)
		s += struct.pack('h',n_active)
		r_str = struct.pack('%sh' % len(r), *r)
		# r_str = reduce(lambda x,y: x+y,r_str)
		s += r_str
	with open(filename,'w') as f:
		f.write(s)

def getPredsFromArray(model,data,nFrames,filenames,res_dir,res_ext,freqs,preds_in=False,weight=0.1,offset=0):
	if preds_in:
		preds = data
	else:
		preds = model.predict(data,verbose=1,batch_size=2048)
	pos = 0
	for i in range(len(nFrames)):
		fname = filenames[i][:-4]
		fname = reduce(lambda x,y: x+'/'+y,fname.split('/')[4:])
		stdout.write("\r%d/%d 	" % (i,len(filenames)))
		stdout.flush()
		res_file_path = res_dir+fname+res_ext
		dirname = os.path.dirname(res_file_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		# preds = model.predict(data[pos:pos+nFrames[i]],batch_size=nFrames[i])
		writeSenScores(res_file_path,preds[pos:pos+nFrames[i]],freqs,weight,offset)
		pos += nFrames[i]

def getPredsFromFilelist(model,filelist,file_dir,file_ext,
							res_dir,res_ext,freqs,context_len=4,
							weight=1,offset=0):
	with open(filelist) as f:
		files = f.readlines()
		files = map(lambda x: x.strip(),files)
	filepaths = map(lambda x: file_dir+x+file_ext,files)
	scaler = StandardScaler(copy=False,with_std=False)
	for i in range(len(filepaths)):
		stdout.write("\r%d/%d 	" % (i,len(filepaths)))
		stdout.flush()

		f = filepaths[i]
		if not os.path.exists(f):
			print "\n",f
			continue
		data = np.loadtxt(f)
		data = scaler.fit_transform(data)

		pad_top = np.zeros((context_len,data.shape[1])) + data[0]
		pad_bot = np.zeros((context_len,data.shape[1])) + data[-1]
		padded_data = np.concatenate((pad_top,data),axis=0)
		padded_data = np.concatenate((padded_data,pad_bot),axis=0)

		data = []
		for j in range(context_len,len(padded_data) - context_len):
			new_row = padded_data[j - context_len: j + context_len + 1]
			new_row = new_row.flatten()
			data.append(new_row)
		data = np.array(data)
		preds = model.predict(data,batch_size=data.shape[0])
		
		res_file_path = res_dir+files[i]+res_ext
		dirname = os.path.dirname(res_file_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		writeSenScores(res_file_path,preds,freqs,weight,offset)