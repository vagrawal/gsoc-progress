import os
CUDA_VISIBLE_DEVICES = '3'
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
	AveragePooling2D, 
	Reshape, 
	Flatten,
	Masking)
from keras.layers.core import Lambda
from keras.layers.merge import add, concatenate
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
from keras.callbacks import History,ModelCheckpoint,CSVLogger,ReduceLROnPlateau
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
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
import ctc
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot

def mlp1(input_dim,output_dim,depth,width,dropout=False,
	BN=False, regularize=False, lin_boost=False):
	print locals()
	model = Sequential()
	model.add(Dense(width, activation='sigmoid', input_shape=(input_dim,),
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
		model.add(Dense(output_dim, name='out'))
		model.add(Activation('softmax', name='softmax'))
	opt = Adam(lr=10/(np.sqrt(input_dim * width * output_dim)))
	model.compile(optimizer=opt,
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def get(t,x,y):
	return K.gather(K.gather(t,x),y)
def get_in_alpha(a,t,s):
	idxs,vals = a
	print idxs
	idx = idxs.index([t,s])
	return vals[idx]

def calc_alpha(p,input_length,label_length,symbols):
	alpha = []
	z = tf.Variable(tf.zeros_like(p[0]))
	target = K.gather(p[0], K.gather(symbols,0))
	z = tf.scatter_update(z, K.gather(symbols,0), target)
	alpha.append(z)
	return alpha
def ctc_loss(args):
	y_pred, labels, input_length, label_length = args
	label_length = K.cast(tf.squeeze(label_length), 'int32')
	input_length = K.cast(tf.squeeze(input_length), 'int32')
	alpha = []
	
	t_arr = tf.TensorArray(tf.float32,1,dynamic_size=True,infer_shape=True)
	Y = t_arr.unstack(y_pred)
	
	t_arr = tf.TensorArray(tf.int32,1,dynamic_size=True,infer_shape=False)
	T = t_arr.unstack(input_length)
	
	t_arr = tf.TensorArray(tf.int32,1,dynamic_size=True,infer_shape=False)
	S = t_arr.unstack(label_length)
	
	t_arr = tf.TensorArray(tf.int32,1,dynamic_size=True,infer_shape=False)
	L = t_arr.unstack(labels)

	b = tf.constant(0,dtype='int32')
	t = tf.constant(0,dtype='int32')
	s = tf.constant(0,dtype='int32')

	while_condition_Y = lambda x: K.less(x,Y.size())
	while_condition_S = lambda x: K.less(x,S.size())
	z = tf.Variable(tf.zeros_like(tf.unstack(Y.read(0))[0]))
	def calc_alpha(p,input_length,label_length,symbols):
		alpha = []
		target = K.gather(p[0], K.gather(symbols,0))		
		alpha.append(tf.scatter_update(z, K.gather(symbols,0), target))

		while_condition_T = lambda x: K.less(x,T.size())
		new_row = z

		return alpha

	def body_Y(b):
		alpha.append(calc_alpha(tf.unstack(Y.read(b)), T.read(b), S.read(b), L.read(b)))
		return [tf.add(b,1)]
	tf.while_loop(while_condition_Y,body_Y,[b],back_prop=False)
	# tf.map_fn(lambda p: calc_alpha(tf.unstack(p),alpha,input_length,label_length,labels), y_pred, back_prop=False)
	print alpha
	labels = K.ctc_label_dense_to_sparse(labels,label_length)
	return tf.nn.ctc_loss(labels,y_pred,input_length,
    				preprocess_collapse_repeated=False,
    				ctc_merge_repeated=True,
    				time_major=False,
    				ignore_longer_outputs_than_inputs=True)

def ctc_lambda_func(args):
	import tensorflow as tf
	y_pred, labels, input_length, label_length = args
	label_length = K.cast(tf.squeeze(label_length), 'int32')
	input_length = K.cast(tf.squeeze(input_length), 'int32')
    # return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	labels = K.ctc_label_dense_to_sparse(labels,label_length)
	return tf.nn.ctc_loss(labels,y_pred,input_length,
    				preprocess_collapse_repeated=False,
    				ctc_merge_repeated=True,
    				time_major=False,
    				ignore_longer_outputs_than_inputs=True)

def ler(y_true, y_pred, **kwargs):
    """
        Label Error Rate. For more information see 'tf.edit_distance'
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))

def decode_output_shape(inputs_shape):
    y_pred_shape, seq_len_shape = inputs_shape
    return (y_pred_shape[:1], None)

def decode(args):
	import tensorflow as tf
	y_pred, label_len = args
	label_len = K.cast(tf.squeeze(label_len), 'int32')
	# ctc_labels = tf.nn.ctc_greedy_decoder(y_pred, label_len)[0][0]
	# return ctc_labels
	ctc_labels = K.ctc_decode(y_pred,label_len,greedy=False)[0][0]
	return K.ctc_label_dense_to_sparse(ctc_labels, label_len)

def ler(args):
	y_pred, y_true, input_length, label_length = args
	label_length = K.cast(tf.squeeze(label_length), 'int32')
	input_length = K.cast(tf.squeeze(input_length), 'int32')

	y_pred = K.ctc_decode(y_pred,input_length)[0][0]
	# y_pred = tf.nn.ctc_greedy_decoder(y_pred,input_length)[0][0]
	y_pred = K.cast(y_pred,'int32')
	# y_pred = math_ops.to_int32(y_pred)
	# y_true = math.ops.to_int64(y_true)
	y_true = K.ctc_label_dense_to_sparse(y_true,label_length)
	y_pred = K.ctc_label_dense_to_sparse(y_pred,input_length)
	return tf.reduce_mean(tf.edit_distance(y_pred, y_true))
def ler(y_true, y_pred, **kwargs):
    """
        Label Error Rate. For more information see 'tf.edit_distance'
    """
    return tf.reduce_mean(tf.edit_distance(y_pred, y_true, **kwargs))
def dummy_loss(y_true,y_pred):
	return y_pred
def decoder_dummy_loss(y_true,y_pred):
	return K.zeros((1,))
def mlp_wCTC(input_dim,output_dim,depth,width,BN=False):
	print locals()
	x = Input(name='x', shape=(1000,input_dim))
	h = x
	h = Masking()(h)
	for i in range(depth):
		h = Dense(width)(h)
		if BN:
			h = BatchNormalization()(h)
		h = Activation('sigmoid')(h)
	out = Dense(output_dim,name='out')(h)
	softmax = Activation('softmax', name='softmax')(out)
	# a = 1.0507 * 1.67326
	# b = -1
	# # out = Lambda(lambda x : a * K.pow(x,3) + b)(h)
	# out = Lambda(lambda x: a * K.exp(x) + b, name='out')(h)
	y = Input(name='y',shape=[None,],dtype='int32')
	x_len = Input(name='x_len', shape=[1],dtype='int32')
	y_len = Input(name='y_len', shape=[1],dtype='int32')

	dec = Lambda(decode, output_shape=decode_output_shape, name='decoder')([out,x_len])
	# edit_distance = Lambda(ler, output_shape=(1,), name='edit_distance')([out,y,x_len,y_len])

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([out, y, x_len, y_len])
	model = Model(inputs=[x, y, x_len, y_len], outputs=[loss_out,dec,softmax])

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.99, nesterov=True, clipnorm=5)
	opt = Adam(lr=0.0001, clipnorm=5)
	model.compile(loss={'ctc': dummy_loss,
						'decoder': decoder_dummy_loss,
						'softmax': 'sparse_categorical_crossentropy'},
					optimizer=opt,
					metrics={'decoder': ler,
							 'softmax': 'accuracy'},
					loss_weights=[1,0,0])
	return model
def ctc_model(model):
	x = model.get_layer(name='x').input

	out = model.get_layer(name='out').output
	softmax = Activation('softmax', name='softmax')(out)

	y = Input(name='y',shape=[1000],dtype='int32')
	x_len = Input(name='x_len', shape=[1],dtype='int32')
	y_len = Input(name='y_len', shape=[1],dtype='int32')

	dec = Lambda(decode, output_shape=decode_output_shape, name='decoder')([out,x_len])
	# edit_distance = Lambda(ler, output_shape=(1,), name='edit_distance')([out,y,x_len,y_len])

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([out, y, x_len, y_len])
	model = Model(inputs=[x, y, x_len, y_len], outputs=[loss_out,dec,softmax])

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.99, nesterov=True, clipnorm=5)
	opt = Adam(lr=0.0001, clipnorm=5)
	model.compile(loss={'ctc': dummy_loss,
						'decoder': decoder_dummy_loss,
						'softmax': 'sparse_categorical_crossentropy'},
					optimizer=opt,
					metrics={'decoder': ler,
							 'softmax': 'accuracy'},
					loss_weights=[1,0,0])
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
	inp = Input(shape=(input_dim,), name='x')
	x = inp
	if conv:
		x = Reshape((11,input_dim/11,1))(x)
		for i in range(nConv):
			print i
			x = LocallyConnected2D(84,(6,8),padding='valid')(x)
			x = _bn_relu(x)
			x = MaxPooling2D((6,6),strides=(1,1),padding='same')(x)
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
		z = Dense(output_dim, name='out')(x)
		z = Activation('softmax', name='softmax')(z)
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
	model = builder.build_resnet_18((input_dim,), output_dim, Reshape((11,input_dim/11,1)))
	opt = Adam(lr=10/np.sqrt(input_dim * output_dim))
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
	if len(inp.shape) == 3:
		inp = inp.reshape((inp.shape[0] * inp.shape[1],inp.shape[2]))
	sigma = np.std(inp)
	# sigma = 1
	rbm = GBRBM(n_visible=inp.shape[-1],n_hidden=width,learning_rate=0.002, momentum=0.90, use_tqdm=True,sample_visible=True,sigma=sigma)
	rbm.fit(inp,n_epoches=15,batch_size=batch_size,shuffle=True)
	RBMs.append(rbm)
	for i in range(depth - 1):
		print 'training DBN layer', i
		rbm = BBRBM(n_visible=width,n_hidden=width,learning_rate=0.02, momentum=0.90, use_tqdm=True)
		for e in range(nEpoches):
			batch_size *= 1 + (e*0.5)
			n_batches = (inp.shape[-2] / batch_size) + (1 if inp.shape[-2]%batch_size != 0 else 0)
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
# def gen_data(active):

def gen_data(alldata,alllabels,batch_size):
	while 1:
		for i in range(batch_size,alldata.shape[0]+1,batch_size):
			x = alldata[i-batch_size:i]
			y = alllabels[i-batch_size:i].reshape(batch_size,alllabels.shape[1])
			y_len = []
			for b in y:
				# print b[-1], int(b[-1]) != 138
				pad_len = 0
				while pad_len < len(b) and int(b[pad_len]) != 138:
					pad_len += 1
				y_len.append(pad_len)
			y_len = np.array(y_len)
			x_len = []
			for b in x:
				# print b[-1], int(b[-1]) != 138
				pad_len = 0
				while pad_len < len(b) and b[pad_len].any():
					pad_len += 1
				x_len.append(pad_len)
			x_len = np.array(x_len)
			# x_len = np.array(map(lambda x: len(x), x))
			# y_len = np.array(map(lambda x: len(x), y))
			# print x.shape,y.shape,x_len,y_len
			# print y.shape
			# y = dense2sparse(y)
			inputs = {'x': x,
					'y': y,
					'x_len': x_len,
					'y_len': y_len}
			outputs = {'ctc': np.ones([batch_size]),
						'decoder': dense2sparse(y),
						'softmax': y.reshape(batch_size,y.shape[1],1)}
			yield(inputs,outputs)
		

def preTrain(model,modelName,x_train,y_train,meta,skip_layers=[],outEqIn=False):
	print model.summary()
	layers = model.layers
	output = layers[-1]
	outdim = output.output_shape[-1]
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
				init_epoch=0, fit_generator=None, ctc_train=False):
	print 'TRAINING MODEL:',modelName
	if not testOnly:
		if pretrain:
			print 'pretraining model...'
			model = preTrain(model,modelName,x_train,y_train,meta)
		if ctc_train:
			model = ctc_model(model)
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
			history = model.fit_generator(fit_generator(x_train,y_train,batch_size),
											steps_per_epoch=x_train.shape[0]/batch_size, epochs=75,
											validation_data=fit_generator(x_test,y_test,batch_size),
											validation_steps = x_test.shape[0],
											callbacks=callback_arr)
		model = Model(inputs=[model.get_layer(name='x').input],
						outputs=[model.get_layer(name='softmax').output])
		print model.summary()
		model.compile(loss='sparse_categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])
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
	meta = np.load('wsj0_phonelabels_bracketed_meta.npz')
	x_train = np.load('wsj0_phonelabels_bracketed_train.npy',mmap_mode='r')
	# train_active = np.load('wsj0_phonelabels_bracketed_train_active.npy')
	y_train = np.load('wsj0_phonelabels_bracketed_train_labels.npy')
	print x_train.shape
	print y_train.shape
	# end = x_train.shape[0] % 2048
	# x_train = x_train[:-end]
	# y_train = y_train[:-end]
	nClasses = int(np.max(y_train)) + 1
	print nClasses
	# print 'transforming labels...'
	# y_train = to_categorical(y_train, num_classes = nClasses)

	print 'loading test data...'
	x_test = np.load('wsj0_phonelabels_bracketed_dev.npy',mmap_mode='r')
	# test_active = np.load('wsj0_phonelabels_bracketed_dev_active.npy')
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
	# model = resnet_wrapper(x_train.shape[-1],nClasses)
	# model = mlp4(x_train.shape[-1], nClasses,2,10,2048,shortcut=True,BN=True,conv=True,dropout=False,regularize=True)
	# model = mlp1(x_train.shape[-1], nClasses,3,2048,BN=True,regularize=False,lin_boost=False)
	# model = mlp_wCTC(x_train.shape[-1],nClasses,3,2048,BN=True)
	model = DBN_DNN(x_train, nClasses,5,3072,batch_size=128)
	# print 'wrapping ctc...'
	# model = ctc_model(model)
	# model = DBN_DNN(x_train, nClasses,5,2560,batch_size=128)
	# model = load_model('mlp4-2x2560-cd-adam-bn-drop-conv-noshort_CP.h5')
	# fg = gen_bracketed_data(context_len=5,for_CTC=True,fix_length=True)
	trainNtest(model,x_train,y_train,x_test,y_test,meta,'dbn-5x3072',ctc_train=False)


	# model = load_model('mlp4_1x2048-conv-BN_CP.h5',custom_objects={'dummy_loss':dummy_loss, 
	# 													'decoder_dummy_loss':decoder_dummy_loss,
	# 													'ler':ler})
	# # model = Model(inputs=[model.get_layer(name='x').input],
	# # 					outputs=[model.get_layer(name='softmax').output])
	# print model.summary()
	# # # # getPredsFromFilelist(model,'../wsj/wsj0/single_dev.txt','/home/mshah1/wsj/wsj0/feat_cd_mls/','.mls','/home/mshah1/wsj/wsj0/single_dev_NN/','.sen',meta['state_freq_Train'],context_len=5,weight=0.00035457)
	# # # # getPredsFromFilelist(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_ci_mls/','.mfc','/home/mshah1/wsj/wsj0/senscores_dev2/','.sen',meta['state_freq_Train'])
	# getPredsFromFilelist(model,'../wsj/wsj0/etc/wsj0_dev.fileids','/home/mshah1/wsj/wsj0/feat_cd_mls/','.mls','/home/mshah1/wsj/wsj0/senscores_dev_conv_ci/','.sen',meta['state_freq_Train'],
	# 						context_len=5, 
	# 						# data_preproc_fn = lambda x: pad_sequences([x],maxlen=1000,dtype='float32',padding='post').reshape(1,1000,x.shape[-1]),
	# 						# data_postproc_fn = lambda x: x[:,range(138)] / np.sum(x[:,range(138)], axis=1).reshape(-1,1),
	# 						weight=0.1*0.01284038)
	# *0.00311573
	# 00269236
	# # getPredsFromArray(model,np.load('DEV_PRED.npy'),meta['framePos_Dev'],meta['filenames_Dev'],'/home/mshah1/wsj/wsj0/senscores_dev_ci_hammad/','.sen',meta['state_freq_Train'],preds_in=True,weight=-0.00075526,offset=234.90414376)
	# f = filter(lambda x : '22go0208.wv1.flac' in x, meta['filenames_Dev'])[0]
	# file_idx = list(meta['filenames_Dev']).index(f)
	# print file_idx
	# # split = lambda x: x[sum(meta['framePos_Dev'][:file_idx]):sum(meta['framePos_Dev'][:file_idx+1])]
	# pred = model.predict(x_test[file_idx:file_idx+1],verbose=1)
	# pred = np.array(map(lambda x: np.argmax(x,axis=-1),pred))
	# print pred
	# print y_test[file_idx]
	# data = split(x_test)
	# context_len = 5
	# pad_top = np.zeros((context_len,data.shape[1]))
	# pad_bot = np.zeros((context_len,data.shape[1]))
	# padded_data = np.concatenate((pad_top,data),axis=0)
	# padded_data = np.concatenate((padded_data,pad_bot),axis=0)

	# data = []
	# for j in range(context_len,len(padded_data) - context_len):
	# 	new_row = padded_data[j - context_len: j + context_len + 1]
	# 	new_row = new_row.flatten()
	# 	data.append(new_row)
	# data = np.array(data)
	# pred = model.predict(data,verbose=1)
	# pred = pred.reshape(1,pred.shape[0],pred.shape[1])
	# print pred.shape
	# [bp],out = K.ctc_decode(pred,[pred.shape[1]])
	# print K.eval(bp), len(K.eval(bp)[0])
	# print K.eval(out)

	# print pred
	# # writeSenScores('senScores',pred)
	# np.save('pred.npy',np.log(pred)/np.log(1.001))

	# plotFromCSV('mlp4-2x2560-cd-adam-bn-drop-conv-noshort')