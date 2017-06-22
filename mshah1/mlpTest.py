import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Activation, Dropout, add, Conv1D, MaxPooling1D, Reshape, Flatten
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
from keras.callbacks import History
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
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot
def mlp1(input_dim,output_dim,depth,width,dropout=False,BN=False):
	model = Sequential()
	model.add(Dense(width, activation='sigmoid', input_dim=input_dim))
	for i in range(depth):
		model.add(Dense(width, activation='sigmoid'))
		if dropout:
			model.add(Dropout(0.25))
		if BN:
			model.add(BatchNormalization())
	model.add(Dense(output_dim, activation='softmax'))
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
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

def make_dense_res_block(inp, size, width):
	x = inp
	for i in range(size):
		x = Dense(width, activation='relu')(x)
		x = Dropout(0.25)(x)
	return x

def mlp4(input_dim,output_dim,nBlocks,width):
	inp = Input(shape=(input_dim,))
	# x = Reshape((input_dim,1))(inp)
	# x = Conv1D(64,4,padding='same',activation='relu')(x)
	# x = MaxPooling1D(3,padding='same')(x)
	# x = Flatten()(x)
	x = Dense(width, activation='relu')(inp)
	for i in range(nBlocks):
		y = make_dense_res_block(x,2,width)
		x = add([x,y])
	z = Dense(output_dim, activation='softmax')(x)
	model = Model(inputs=inp, outputs=z)
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def preTrain(model,x_train,y_train,skip_layers=[],weights=None):
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
		last = model.layers[i].output
		preds = Dense(outdim,activation='softmax')(last)
		model_new = Model(model.input,preds)
		for j in range(1,i+1):
			print "untrainable layer ",j
			model_new.layers[j].trainable=False
		model_new.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
		print model_new.summary()
		model_new.fit(x_train,y_train,epochs=3,batch_size=20000)
		model.layers[i].set_weights(model_new.layers[-2].get_weights())
	for l in model.layers:
		l.trainable = True
	return model

def trainDBN_DNN(x_train,nClasses,depth,width):
	print 'loading data...'
	inp = x_train
	input_dim = inp.shape[1]
	weights = []
	bias = []
	batch_size = 150000
	for i in range(depth):
		n_batches = (inp.shape[0] / batch_size)
		out = np.memmap('temp_arr.dat',dtype='float16',shape=((n_batches/2)*batch_size,width),mode='w+')
		print 'training DBN layer', i
		if i == 0:
			rbm = GBRBM(n_visible=input_dim,n_hidden=width,learning_rate=0.01, momentum=0.95, use_tqdm=True)
			rbm.fit(inp,n_epoches=1,batch_size=20000,shuffle=False)
			gc.collect()
			batch_idx = np.random.choice(n_batches,size=n_batches/2,replace=False)
		else:
			rbm = BBRBM(n_visible=input_dim,n_hidden=width,learning_rate=0.01, momentum=0.95, use_tqdm=True)
			rbm.fit(inp,n_epoches=1,batch_size=20000,shuffle=False)
			gc.collect()
			batch_idx = xrange(n_batches)

		(W,_,Bh) = rbm.get_weights()
		weights.append(W)
		bias.append(Bh)

		print 'batch transforming data...'
		for j in range(len(batch_idx)):
			idx = batch_idx[j]
			stdout.write("\r%d batch no %d/%d" % (int(time.time()),j+1,len(batch_idx)))
			stdout.flush()
			b = np.array(inp[idx*batch_size:min((idx+1)*batch_size, inp.shape[0])])
			T = rbm.transform(b)
			out[j*batch_size:min((j+1)*batch_size, inp.shape[0])] = T.astype('float16')
		print 'batch transform finished...'
		inp = out
		stdout.write("\n")
		stdout.flush()
		input_dim = inp.shape[1]

	model = mlp1(x_train.shape[1],nClasses,depth-1,width)
	print len(weights), len(model.layers)
	assert len(weights) == len(model.layers) - 1
	for i in range(len(weights)):
		W = [weights[i],bias[i]]
		model.layers[i].set_weights(W)
	return model

def normalizeByUtterance(data,nFrames):
	print 'normalizing...'
	scaler = StandardScaler(copy=False)
	# print data
	pos = 0
	for i in xrange(len(nFrames)):
		stdout.write("\rnormalizing utterance no %d " % i)
		stdout.flush()
		data[pos:nFrames[i]] = scaler.fit_transform(data[pos:nFrames[i]])
		pos = nFrames[i]
	# print data

class TestCallback(History):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_train_begin(self,logs=None):
    	super(TestCallback,self).on_train_begin(logs)
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['eval_loss'] = loss
        logs['eval_acc'] = acc
        super(TestCallback,self).on_epoch_end(epoch, logs)
        print('\nTesting loss: {}, Testing acc: {}\n'.format(loss, acc))

def trainNtest(model,x_train,y_train,x_test,y_test,modelName,plot_name,testOnly=False,preTrain=True):

	# model = mlp4(20, 132,2,2048)
	# print model.summary()
	# plot_model(model, to_file='mlp4_model.png')
	# model = trainDBN_DNN('wsj0_phonelabels_bracketed.npz',4,512)

	# print 'normalizing the data...'
	# scaler = StandardScaler(copy=False)
	# scaler.fit(x_train)
	# x_train = scaler.transform(x_train)
	if not testOnly:
		if preTrain:
			print 'pretraining model...'
			model = preTrain(model,x_train,y_train)
		print model.summary()
		print 'starting fit...'
		history = model.fit(x_train,y_train,epochs=10,batch_size=20000,
							callbacks=[TestCallback((x_test,y_test))])

		print 'saving model...'
		model.save(modelName+'.h5')
		print(history.history.keys())

		print 'plotting graphs...'
		# summarize history for accuracy
		plt.plot(history.history['acc'])
		plt.plot(history.history['eval_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['training acc', 'testing acc'])
		plt.savefig(plot_name+'.png')
		plt.clf()
	else:
		model = load_model(modelName)
	print 'scoring...'
	score = model.evaluate(x_test, y_test, batch_size=2000)
	print score

print 'loading data...'
# meta = np.load('wsj0_phonelabels_bracketed_meta.npz')
x_train = np.load('wsj0_phonelabels_bracketed_train.npy',mmap_mode='r')
y_train = np.load('wsj0_phonelabels_bracketed_train_labels.npy',mmap_mode='r')
nClasses = y_train.shape[1]
# print 'transforming labels...'
# y_train = to_categorical(y_train, num_classes = nClasses)

print 'loading test data...'
x_test = np.load('wsj0_phonelabels_bracketed_dev.npy',mmap_mode='r')
y_test = np.load('wsj0_phonelabels_bracketed_dev_labels.npy',mmap_mode='r')
# print 'transforming labels...'
# y_test = to_categorical(y_test, num_classes = nClasses)

print 'initializing model...'
# model = mlp1(x_train.shape[1], nClasses,2,2048,dropout=True)
model = trainDBN_DNN(x_train, nClasses,3,2048)
trainNtest(model,x_train,y_train,x_test,y_test,'dbn-3x2048-sig-adagrad','dbn-3x2048-sig-adagrad',preTrain=False)
# trainNtest(None,None,None,x_test,y_test,'mlp1-3x2048-sig-adagrad.h5','',testOnly=True)
# t1 = threading.Thread(target=trainNtest,args=(model,x_train,y_train,x_test,y_test,'mlp1-3x2048-sig-adadelta-drop','mlp1-3x2048-sig-adadelta-drop'))
# t1 = threading.Thread(target=trainNtest,args=(m2,x_train,y_train,x_test,y_test,'mlp4-3x2048','mlp4-3x2048'))