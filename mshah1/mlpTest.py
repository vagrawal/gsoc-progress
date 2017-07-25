import os
CUDA_VISIBLE_DEVICES = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adagrad, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import (
	Input, 
	Dense,
	Activation, 
	Dropout, 
	Conv1D, 
	Conv2D,
	LocallyConnected2D, 
	MaxPooling2D, 
	Reshape, 
	Flatten)
from keras.layers.core import Lambda
from keras.layers.merge import add, concatenate
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
from keras.callbacks import History,ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from keras import regularizers
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
from utils import *
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot

def mlp1(input_dim,output_dim,depth,width,dropout=False,
	BN=False, regularize=False, lin_boost=False):
	print locals()
	model = Sequential()
	model.add(Dense(width, activation='sigmoid', input_dim=input_dim,
					kernel_regularizer=regularizers.l2(0.05) if regularize else None))
	if BN:
		model.add(BatchNormalization())
	if dropout:
		model.add(Dropout(0.15))
	for i in range(depth):
		model.add(Dense(width, activation='sigmoid',
						kernel_regularizer=regularizers.l2(0.05) if regularize else None))
		if dropout:
			model.add(Dropout(0.15))
		if BN:
			model.add(BatchNormalization())
	if lin_boost:
		model.add(Dense(output_dim))	
		model.add(Lambda(lambda x: K.exp(x)))
	else:	
		model.add(Dense(output_dim, activation='softmax'))
	opt = Adam(lr=10/(np.sqrt(input_dim * width * output_dim)))
	model.compile(optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def mlp_wCTC(input_dim,output_dim,depth,width,BN=False):
	x = Input(name='x', shape=(1000,input_dim))
	h = Dense(width,activation='sigmoid')(x)
	if BN:
		BatchNormalization()(h)
	for i in range(depth-1):
		h = Dense(width,activation='sigmoid')(h)
		if BN:
			BatchNormalization()(h)
	h = Dense(output_dim)(h)
	out = Activation('softmax', name='softmax')(h)

	y = Input(name='y',shape=[1000])
	x_len = Input(name='x_len', shape=[1])
	y_len = Input(name='y_len', shape=[1])

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([out, y, x_len, y_len])
	model = Model(inputs=[x, y, x_len, y_len], outputs=loss_out)

	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
					optimizer='adam',
					metrics=['accuracy'])

	return model

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def make_dense_res_block(inp, size, width, drop=False,BN=False,regularize=False):
	x = inp
	for i in range(size):
		x = Dense(width,
					kernel_regularizer=regularizers.l2(0.05) if regularize else None)(x)
		if i < size - 1:
			if drop:
				x = Dropout(0.15)(x)
			if BN:
				x = _bn_relu(x)			
	return x

def mlp4(input_dim,output_dim,nConv,nBlocks,width, block_depth=2, 
			block_width=None, dropout=False, BN=False, 
			parallelize=False, conv=False, regularize=False,
			exp_boost=False, quad_boost=False, shortcut=True):
	print locals()
	if block_width == None:
		block_width = width
	inp = Input(shape=(input_dim,))
	if conv:
		x = Reshape((11,input_dim/11,1))(inp)
		for i in range(nConv):
			print i
			x = LocallyConnected2D(84,(8,8),padding='valid')(x)
			x = _bn_relu(x)
			x = MaxPooling2D((6,6),strides=(2,2),padding='same')(x)
		x = Flatten()(x)
		x = Dense(width,
					kernel_regularizer=regularizers.l2(0.05) if regularize else None)(x)
	else:
		x = Dense(width,
					kernel_regularizer=regularizers.l2(0.05) if regularize else None)(inp)
	if dropout:
		x = Dropout(0.15)(x)
	if BN:
		x = _bn_relu(x)
	if block_width != width:
		x = Dense(block_width)(x)
	for i in range(nBlocks):
		y = make_dense_res_block(x,block_depth,block_width,BN=BN,drop=dropout,regularize=regularize)
		if shortcut:
			x = add([x,y])
		else:
			x = y
		if dropout:
			x = Dropout(0.15)(x)
		if BN:
			x = _bn_relu(x)

	if exp_boost:
		x = Dense(output_dim)(x)
		z = Lambda(lambda x : K.exp(x))(x)
	if quad_boost:
		x = Dense(output_dim)(x)
		a = 0.001
		b = 0.4
		z = Lambda(lambda x : a * K.pow(x,3) + b)(x)
	else:
		z = Dense(output_dim, activation='softmax')(x)
	model = Model(inputs=inp, outputs=z)
	if parallelize:
		model = make_parallel(model, len(CUDA_VISIBLE_DEVICES.split(',')))
	opt = Adam(lr=25/(np.sqrt(input_dim * width * output_dim)))
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
	nEpoches = 5

	sigma = np.std(inp)
	# sigma = 1
	rbm = GBRBM(n_visible=inp.shape[1],n_hidden=width,learning_rate=0.002, momentum=0.90, use_tqdm=True,sample_visible=True,sigma=sigma)
	rbm.fit(inp,n_epoches=5,batch_size=batch_size,shuffle=True)
	RBMs.append(rbm)
	for i in range(depth - 1):
		print 'training DBN layer', i
		rbm = BBRBM(n_visible=width,n_hidden=width,learning_rate=0.02, momentum=0.90, use_tqdm=True)
		for e in range(nEpoches):
			batch_size *= 1 + (e*0.5)
			n_batches = (inp.shape[0] / batch_size) + (1 if inp.shape[0]%batch_size != 0 else 0)
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
	re
def gen_data(alldata,alllabels,batch_size):
	n_batches = (alldata.shape[0] / batch_size) + (1 if alldata.shape[0]%batch_size != 0 else 0)
	# nClasses = np.max(alllabels) + 1
	nClasses = 4138
	while 1:
		idxs = range(alldata.shape[0])
		np.random.shuffle(idxs)
		for j in range(n_batches):
			idx = idxs[j*batch_size:min((j+1)*batch_size, len(idxs))]
			data = np.array(map(lambda x: alldata[x],idx))
			labels = np.array(map(lambda x: alllabels[x],idx))
			if len(labels.shape) == 1:
				labels = to_categorical(labels,num_classes=nClasses)
			
			yield (data,labels)

		

def preTrain(model,modelName,x_train,y_train,meta,skip_layers=[],outEqIn=False):
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
		if outEqIn:
			preds = Dense(outdim)(last)
		else:
			preds = Dense(outdim,activation='softmax')(last)
		model_new = Model(model.input,preds)
		for j in range(len(model_new.layers) - 2):
			print "untrainable layer ",j
			model_new.layers[j].trainable=False
		model_new.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
		print model_new.summary()
		batch_size = 2048
		model_new.fit(x_train,y_train,epochs=1,batch_size=2048)
		# model.fit_generator(gen_bracketed_data(x_train,y_train,meta['framePos_Train'],4),
		# 								steps_per_epoch=len(meta['framePos_Train']), epochs=3,
		# 								callbacks=[ModelCheckpoint('%s_CP.h5' % modelName,monitor='loss',mode='min')])
		# model.fit_generator(gen_data(x_train,y_train,batch_size),
		# 					steps_per_epoch = x_train.shape[0] / batch_size,
		# 					epochs = 1)
		model.layers[i].set_weights(model_new.layers[-2].get_weights())
	for l in model.layers:
		l.trainable = True
	return model

def trainNtest(model,x_train,y_train,x_test,y_test,meta,
				modelName,testOnly=False,pretrain=False,
				init_epoch=0, fit_generator=None):
	print 'TRAINING MODEL:',modelName
	if not testOnly:
		if pretrain:
			print 'pretraining model...'
			model = preTrain(model,modelName,x_train,y_train,meta)
		print model.summary()
		print 'starting fit...'
		callback_arr = [ModelCheckpoint('%s_CP.h5' % modelName,save_best_only=True,verbose=1),
						ReduceLROnPlateau(patience=5,factor=0.5,min_lr=10**(-6), verbose=1),
						CSVLogger(modelName+'.csv',append=True)]
		batch_size = 512
		if fit_generator == None:
			history = model.fit(x_train,y_train,epochs=100,batch_size=batch_size,
								initial_epoch=init_epoch,
								validation_data=(x_test,y_test),
								callbacks=callback_arr)
		else:
			history = model.fit_generator(fit_generator(x_train,y_train,meta['framePos_Train']),
											steps_per_epoch=len(meta['framePos_Train']), epochs=30,
											validation_data=fit_generator(x_test,y_test,meta['framePos_Dev']),
											validation_steps = len(meta['framePos_Dev']),
											callbacks=callback_arr)
		print 'saving model...'
		model.save(modelName+'.h5')
		# model.save_weights(modelName+'_W.h5')
		print(history.history.keys())
		print history.history['lr']
		print 'plotting graphs...'
		# summarize history for accuracy
		fig, ax1 = plt.subplots()
		ax1.plot(history.history['acc'])
		ax1.plot(history.history['val_acc'])
		ax2 = ax1.twinx()
		ax2.plot(history.history['loss'],color='r')
		ax2.plot(history.history['val_loss'],color='g')
		plt.title('model loss & accuracy')
		ax1.set_ylabel('accuracy')
		ax2.set_ylabel('loss')
		ax1.set_xlabel('epoch')
		ax1.legend(['training acc', 'testing acc'])
		ax2.legend(['training loss', 'testing loss'])
		fig.tight_layout()
		plt.savefig(modelName+'.png')
		plt.clf()
	else:
		model = load_model(modelName)
		print 'scoring...'
		score = model.evaluate_generator(gen_bracketed_data(x_test,y_test,meta['framePos_Dev'],4),
										len(meta['framePos_Dev']))
		print score

if __name__ == '__main__':
	print 'PROCESS-ID =', os.getpid()
	print 'loading data...'
	meta = np.load('wsj0_phonelabels_meta.npz')
	x_train = np.load('wsj0_phonelabels_train.npy',mmap_mode='r')
	y_train = np.load('wsj0_phonelabels_train_labels.npy')
	# end = x_train.shape[0] % 2048
	# x_train = x_train[:-end]
	# y_train = y_train[:-end]
	nClasses = 139
	print nClasses
	# print 'transforming labels...'
	# y_train = to_categorical(y_train, num_classes = nClasses)

	print 'loading test data...'
	x_test = np.load('wsj0_phonelabels_dev.npy',mmap_mode='r')
	y_test = np.load('wsj0_phonelabels_dev_labels.npy')
	# # # end = x_test.shape[0] % 2048
	# # # x_test = x_test[:-end]
	# # # y_test = y_test[:-end]
	# # # x_test = x_train
	# # # y_test = y_train
	# print 'transforming labels...'
	# y_test = to_categorical(y_test, num_classes = nClasses)

	print 'initializing model...'
	# model = load_model('dbn-3x2048-sig-adagrad_CP.h5')
	# model = mlp4(x_train.shape[1]*11, nClasses,1,2,2560,shortcut=False,BN=True,conv=True,dropout=True,regularize=False)
	# model = mlp1(x_train.shape[1]*11, nClasses,2,2048,BN=True,regularize=False,lin_boost=False)
	model = mlp_wCTC(x_train.shape[1]*11,nClasses,3,2048,BN=True)
	# model = load_model('test_CP.h5')
	# model = DBN_DNN(x_train, nClasses,5,2560,batch_size=128)
	# model = load_model('mlp4-2x2560-cd-adam-bn-drop-conv-noshort_CP.h5')
	fg = gen_bracketed_data(context_len=5,for_CTC=True,fix_length=True)
	trainNtest(model,x_train,y_train,x_test,y_test,meta,'mlp_wCTC-3x2048-ci-adam-bn',fit_generator=fg)


	# model = load_model('newmodel')
	# print model.summary()
	# getPredsFromFilelist(model,'../wsj/wsj0/single_dev.txt','/home/mshah1/wsj/wsj0/feat_cd_mls/','.mls','/home/mshah1/wsj/wsj0/single_dev_NN/','.sen',meta['state_freq_Train'],context_len=5,weight=0.00035457)
	# getPredsFromFilelist(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_ci_mls/','.mfc','/home/mshah1/wsj/wsj0/senscores_dev2/','.sen',meta['state_freq_Train'])
	# getPredsFromFilelist(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_cd_mls/','.mls','/home/mshah1/wsj/wsj0/senscores_dev_cd/','.sen',meta['state_freq_Train'],context_len=5)
	# getPredsFromArray(model,np.load('DEV_PRED.npy'),meta['framePos_Dev'],meta['filenames_Dev'],'/home/mshah1/wsj/wsj0/senscores_dev_ci_hammad/','.sen',meta['state_freq_Train'],preds_in=True,weight=-0.00075526,offset=234.90414376)
	# f = filter(lambda x : '22go0208.wv1.flac' in x, meta['filenames_Dev'])[0]
	# file_idx = list(meta['filenames_Dev']).index(f)
	# # print file_idx
	# split = lambda x: x[sum(meta['framePos_Dev'][:file_idx]):sum(meta['framePos_Dev'][:file_idx+1])]
	# # pred = model.evaluate(split(x_test),split(y_test),verbose=1)
	# pred = model.predict(split(x_test),verbose=1)
	# # print pred
	# # writeSenScores('senScores',pred)
	# np.save('pred.npy',np.log(pred)/np.log(1.001))

	# plotFromCSV('mlp4-2x2560-cd-adam-bn-drop-conv-noshort')