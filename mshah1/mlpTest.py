import os
CUDA_VISIBLE_DEVICES = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Activation, Dropout, Conv1D, Conv2D, MaxPooling2D, Reshape, Flatten
from keras.layers.core import Lambda
from keras.layers.merge import add, concatenate
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
from keras.callbacks import History,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tfrbm import GBRBM,BBRBM
from sys import stdout
import time
import gc
from sklearn.preprocessing import StandardScaler
from guppy import hpy
import threading
import struct
import resnet
# from multi_gpu import make_parallel
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot

def mlp1(input_dim,output_dim,depth,width,dropout=False,BN=False):
	model = Sequential()
	model.add(Dense(width, activation='sigmoid', input_dim=input_dim))
	if BN:
		model.add(BatchNormalization())
	for i in range(depth):
		model.add(Dense(width, activation='sigmoid'))
		if dropout:
			model.add(Dropout(0.25))
		if BN:
			model.add(BatchNormalization())
	model.add(Dense(output_dim, activation='softmax'))
	opt = Adagrad(lr=100/(np.sqrt(input_dim * width * output_dim)))
	model.compile(optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def mlp2(input_dim,output_dim):
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(output_dim, activation='softmax'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	return model

def mlp3(input_dim,output_dim):
	model = Sequential()
	model.add(Dense(3000, activation='relu', input_dim=input_dim))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(output_dim, activation='softmax'))
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def make_dense_res_block(inp, size, width, drop=False,BN=False):
	x = inp
	for i in range(size):
		x = Dense(width)(x)
		if drop:
			x = Dropout(0.5)(x)
		if BN and i < size - 1:
			x = _bn_relu(x)
	return x

def mlp4(input_dim,output_dim,nConv,nBlocks,width, 
			block_width=None, dropout=False, BN=False, 
			parallelize=False, conv=False):
	if block_width == None:
		block_width = width
	inp = Input(shape=(input_dim,))
	if conv:
		x = Reshape((9,input_dim/9,1))(inp)
		for i in range(nConv):
			print i
			x = Conv2D(2**(6+i),(3,8),padding='same')(x)
			x = _bn_relu(x)
			x = MaxPooling2D((2*(i+1),2*(i+1)),padding='same')(x)
		x = Flatten()(x)
		x = Dense(width)(x)
	else:
		x = Dense(width)(inp)
	if BN:
		x = _bn_relu(x)
	for i in range(nBlocks):
		y = make_dense_res_block(x,2,block_width,BN=BN,drop=dropout)
		if block_width != width:
			y = Dense(width)(x)
		x = add([x,y])
		if BN:
			x = _bn_relu(x)
	z = Dense(output_dim, activation='softmax')(x)
	model = Model(inputs=inp, outputs=z)
	if parallelize:
		model = make_parallel(model, len(CUDA_VISIBLE_DEVICES.split(',')))
	opt = Adagrad(lr=1/(np.sqrt(input_dim * width)))
	# opt = SGD(lr=1/(np.sqrt(input_dim * width)), decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def resnet_wrapper(input_dim,output_dim):
	builder = resnet.ResnetBuilder()
	model = builder.build_resnet_18((1,1,input_dim), output_dim, Reshape((9,input_dim/9,1)))
	opt = Adagrad(lr=1/np.sqrt(input_dim))
	# opt = SGD(lr=1/(np.sqrt(input_dim * width)), decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model
def DBN_DNN(inp,nClasses,depth,width,batch_size=2048):
	RBMs = []
	weights = []
	bias = []
	# batch_size = inp.shape
	nEpoches = 75
	n_batches = (inp.shape[0] / batch_size) + (1 if inp.shape[0]%batch_size != 0 else 0)

	sigma = np.std(inp)
	# sigma = 1
	rbm = GBRBM(n_visible=inp.shape[1],n_hidden=width,learning_rate=0.002, momentum=0.90, use_tqdm=True,sample_visible=True,sigma=sigma)
	rbm.fit(inp,n_epoches=100,batch_size=batch_size,shuffle=True)
	RBMs.append(rbm)
	for i in range(depth - 1):
		print 'training DBN layer', i
		rbm = BBRBM(n_visible=width,n_hidden=width,learning_rate=0.02, momentum=0.90, use_tqdm=True)
		for e in range(nEpoches):
			for j in range(n_batches):
				stdout.write("\r%d batch no %d/%d epoch no %d/%d" % (int(time.time()),j+1,n_batches,e,nEpoches))
				stdout.flush()
				b = np.array(inp[j*batch_size:min((j+1)*batch_size, inp.shape[0])])
				for r in RBMs:
					b = r.transform(b)
				rbm.partial_fit(b)
		RBMs.append(rbm)
	for r in RBMs:
		(W,_,Bh) = r.get_weights()
		weights.append(W)
		bias.append(Bh)
	model = mlp1(x_train.shape[1],nClasses,depth-1,width)
	print len(weights), len(model.layers)
	assert len(weights) == len(model.layers) - 1
	for i in range(len(weights)):
		W = [weights[i],bias[i]]
		model.layers[i].set_weights(W)
	return model

def processByUtterance(data,nFrames,fn):
	print 'processing...'
	pos = 0
	for i in xrange(len(nFrames)):
		stdout.write("\rnormalizing utterance no %d " % i)
		stdout.flush()
		data[pos:pos + nFrames[i]] = fn(data[pos:pos + nFrames[i]])
		pos += nFrames[i]
	# print data


def gen_bracketed_data(alldata,alllabels,nFrames,context_len):
	batch_size = 512
	while 1:
		pos = 0
		nClasses = np.max(alllabels) + 1
		for i in xrange(len(nFrames)):
			data = alldata[pos:pos + nFrames[i]]
			labels = alllabels[pos:pos + nFrames[i]]

			if len(labels.shape) == 1:
				labels = to_categorical(labels,num_classes=nClasses)
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
			# if data.shape[0] < batch_size:
			# 	pad_bot = np.zeros((batch_size-data.shape[0],data.shape[1])) + data[-1]
			# 	data = np.concatenate((data,pad_bot),axis=0)
			# 	pad_bot = np.zeros((batch_size-labels.shape[0],labels.shape[1])) + labels[-1]
			# 	labels = np.concatenate((labels,pad_bot),axis=0)
			# 	yield (data,labels)
			# if data.shape[0] > batch_size:
			# 	n_batches = data.shape[0] / batch_size + int(data.shape[0] % batch_size != 0)
			# 	for j in range(n_batches):
			# 		data = np.array(data[j*batch_size:(j+1)*batch_size])
			# 		labels = np.array(labels[j*batch_size:(j+1)*batch_size])
			# 		yield (data,labels)
			yield (data,labels)
			pos += nFrames[i]

class TestCallback(History):
    def __init__(self, test_data, nFrames):
        self.test_data = test_data
        self.nFrames = nFrames
    def on_train_begin(self,logs=None):
    	super(TestCallback,self).on_train_begin(logs)
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate_generator(gen_bracketed_data(x,y,self.nFrames,4),
										len(self.nFrames))
        logs['eval_loss'] = loss
        logs['eval_acc'] = acc
        super(TestCallback,self).on_epoch_end(epoch, logs)
        print('\nTesting loss: {}, Testing acc: {}\n'.format(loss, acc))


def preTrain(model,modelName,x_train,y_train,meta,skip_layers=[],train_layer_0=False):
	print model.summary()
	layers = model.layers
	output = layers[-1]
	outdim = output.output_shape[1]
	for i in range(len(layers) - 1):
		if i in skip_layers:
			print 'skipping layer ',i
			continue
		if len(model.layers[i].get_weights()) == 0:
			print 'skipping layer ',i
			continue
		last = model.layers[i].get_output_at(-1)
		preds = Dense(outdim,activation='softmax')(last)
		model_new = Model(model.input,preds)
		for j in range(len(model_new.layers) - 2):
			print "untrainable layer ",j
			model_new.layers[j].trainable=False
		model_new.compile(optimizer='adagrad',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
		print model_new.summary()
		model_new.fit(x_train,y_train,epochs=1,batch_size=256)
		# model.fit_generator(gen_bracketed_data(x_train,y_train,meta['framePos_Train'],4),
		# 								steps_per_epoch=len(meta['framePos_Train']), epochs=3,
		# 								callbacks=[ModelCheckpoint('%s_CP.h5' % modelName,monitor='loss',mode='min')])
		model.layers[i].set_weights(model_new.layers[-2].get_weights())
	for l in model.layers:
		l.trainable = True
	return model

def trainNtest(model,x_train,y_train,x_test,y_test,meta,modelName,plot_name,testOnly=False,pretrain=False,init_epoch=0):

	# model = mlp4(20, 132,2,2048)
	# print model.summary()
	# plot_model(model, to_file='mlp4_model.png')
	# model = trainDBN_DNN('wsj0_phonelabels_bracketed.npz',4,512)

	# print 'normalizing the data...'
	# scaler = StandardScaler(copy=False)
	# scaler.fit(x_train)
	# x_train = scaler.transform(x_train)
	if not testOnly:
		if pretrain:
			print 'pretraining model...'
			model = preTrain(model,modelName,x_train,y_train,meta)
		print model.summary()
		print 'starting fit...'

		history = model.fit(x_train,y_train,epochs=50,batch_size=2048,
							validation_data=(x_test,y_test),
							callbacks=[ModelCheckpoint('%s_CP.h5' % modelName,save_best_only=True,mode='min'),
										ReduceLROnPlateau(patience=5,min_lr=10*(-6))])
		# EarlyStopping(monitor='val_acc',min_delta=0.25,patience=1,mode='max')
		# history = model.fit_generator(gen_bracketed_data(x_train,y_train,meta['framePos_Train'],4),
		# 								steps_per_epoch=len(meta['framePos_Train']), epochs=30,
		# 								validation_data=gen_bracketed_data(x_test,y_test,meta['framePos_Dev'],4),
		# 								validation_steps = len(meta['framePos_Dev']),
		# 								callbacks=[ModelCheckpoint('%s_CP.h5' % modelName,mode='min')])
		print 'saving model...'
		model.save(modelName+'.h5')
		print(history.history.keys())

		print 'plotting graphs...'
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['training acc', 'testing acc'])
		plt.savefig(plot_name+'.png')
		plt.clf()
	else:
		model = load_model(modelName)
		print 'scoring...'
		score = model.evaluate_generator(gen_bracketed_data(x_test,y_test,meta['framePos_Dev'],4),
										len(meta['framePos_Dev']))
		print score

def writeSenScores(filename,scores,freqs):
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
	scores /= freqs + (1.0 / len(freqs))
	scores = np.log(scores)/np.log(1.0001)
	scores *= -1
	scores -= np.min(scores,axis=1).reshape(-1,1)
	# scores = scores.astype(int)
	scores *= 0.1 * 0.00282001
	scores += 271.10506735
	truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
	vf = np.vectorize(truncateToShort)
	scores = vf(scores)
	# scores /= np.sum(scores,axis=0)
	for r in scores:
		print np.argmin(r)
		s += struct.pack('h',n_active)
		r_str = struct.pack('%sh' % len(r), *r)
		# r_str = reduce(lambda x,y: x+y,r_str)
		s += r_str
	with open(filename,'w') as f:
		f.write(s)

def getPreds(model,filelist,file_dir,file_ext,res_dir,res_ext,freqs,context_len=4):
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
		preds = model.predict(data)
		res_file_path = res_dir+files[i]+res_ext
		dirname = os.path.dirname(res_file_path)
		if not os.path.exists(dirname):
			os.makedirs(dirname)
		writeSenScores(res_file_path,preds,freqs)

print 'loading data...'
meta = np.load('wsj0_phonelabels_bracketed_meta.npz')
x_train = np.load('wsj0_phonelabels_bracketed_train.npy',mmap_mode='r')
y_train = np.load('wsj0_phonelabels_bracketed_train_labels.npy')
# end = x_train.shape[0] % 2048
# x_train = x_train[:-end]
# y_train = y_train[:-end]
nClasses = np.max(y_train) + 1
print nClasses
# print 'transforming labels...'
# y_train = to_categorical(y_train, num_classes = nClasses)

print 'loading test data...'
x_test = np.load('wsj0_phonelabels_bracketed_dev.npy',mmap_mode='r')
y_test = np.load('wsj0_phonelabels_bracketed_dev_labels.npy')
# # # end = x_test.shape[0] % 2048
# # # x_test = x_test[:-end]
# # # y_test = y_test[:-end]
# # # x_test = x_train
# # # y_test = y_train
# print 'transforming labels...'
# y_test = to_categorical(y_test, num_classes = nClasses)

print 'initializing model...'
# model = load_model('dbn-3x2048-sig-adagrad_CP.h5')
model = mlp4(x_train.shape[1], nClasses,3,10,2048,BN=True,conv=False,dropout=True)
# model = mlp1(x_train.shape[1], nClasses,8,2048,BN=True)
# model = load_model('test_CP.h5')
# model = DBN_DNN(x_train, nClasses,5,2048,batch_size=128)
# # model = resnet_wrapper(x_train.shape[1], nClasses)
trainNtest(model,x_train,y_train,x_test,y_test,meta,'mlp4-10x2048-adagrad-drop-cd','mlp4-10x2048-adagrad-drop-cd')



# model = load_model('mlp1-3x2048-sig-adagrad-cd.h5')
# getPreds(model,'../wsj/wsj0/single_dev.txt','/home/mshah1/wsj/wsj0/feat_ci_mls/','.mfc','/home/mshah1/wsj/wsj0/single_dev_NN/','.sen',meta['state_freq_Train'])
# getPreds(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_ci_mls/','.mfc','/home/mshah1/wsj/wsj0/senscores_dev/','.sen',meta['state_freq_Train'])
# getPreds(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_cd_mls/','.mls','/home/mshah1/wsj/wsj0/senscores_dev_cd/','.sen',meta['state_freq_Train'])
# f = filter(lambda x : '22go0208.wv1.flac' in x, meta['filenames_Dev'])[0]
# file_idx = list(meta['filenames_Dev']).index(f)
# # print file_idx
# split = lambda x: x[sum(meta['framePos_Dev'][:file_idx]):sum(meta['framePos_Dev'][:file_idx+1])]
# # pred = model.evaluate(split(x_test),split(y_test),verbose=1)
# pred = model.predict(split(x_test),verbose=1)
# # print pred
# # writeSenScores('senScores',pred)
# np.save('pred.npy',np.log(pred)/np.log(1.001))