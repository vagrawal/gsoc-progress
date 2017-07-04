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
from keras.callbacks import History,ModelCheckpoint,EarlyStopping
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
# from multi_gpu import make_parallel
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot

def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

def make_parallel(model, gpu_count):
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))
            
        return Model(input=model.inputs, output=merged)


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
	opt = Adagrad(lr=1/(np.sqrt(input_dim) * np.sqrt(width)))
	model.compile(optimizer=opt,
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
			x = Dropout(0.25)(x)
		if BN and i < size - 1:
			x = _bn_relu(x)
	return x

def mlp4(input_dim,output_dim,nBlocks,width, 
			block_width=None, drop=False, BN=False, 
			parallelize=False, conv=False):
	if block_width == None:
		block_width = width
	inp = Input(shape=(input_dim,))
	if conv:
		x = Reshape((9,input_dim/9,1))(inp)
		x = Conv2D(64,(4,4),padding='valid',activation='relu')(x)
		x = MaxPooling2D(3,padding='valid')(x)
		x = Flatten()(x)
		x = Dense(width)(x)
	else:
		x = Dense(width)(inp)
	if BN:
		x = _bn_relu(x)
	for i in range(nBlocks):
		y = make_dense_res_block(x,2,block_width,BN=BN,drop=drop)
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
	              loss='categorical_crossentropy',
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
				print "\r%d batch no %d/%d epoch no %d/%d" % (int(time.time()),j+1,n_batches,e,nEpoches)
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
	              loss='categorical_crossentropy',
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
							callbacks=[ModelCheckpoint('%s_CP.h5' % modelName,mode='min')])
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

def writeSenScores(filename,scores):
	n_active = scores.shape[1]
	s = """s3
version 0.1
mdef_file ../../en_us.ci_cont/mdef
n_sen 138
logbase 1.000300
endhdr
"""
	s += struct.pack('l',0x11223344)
	scores = np.log(scores)/np.log(1.0003)
	truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
	vf = np.vectorize(truncateToShort)
	scores = vf(scores)
	scores -= np.max(scores)
	scores /= np.sum(scores)
	for r in scores:
		s += struct.pack('h',n_active)
		r_str = struct.pack('%sh' % len(r), *r)
		# r_str = reduce(lambda x,y: x+y,r_str)
		s += r_str
	with open(filename,'w') as f:
		f.write(s)

def getPreds(model,filelist,file_dir,file_ext,res_dir,res_ext,context_len=4):
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
		writeSenScores(res_file_path,preds)

print 'loading data...'
meta = np.load('wsj0_phonelabels_bracketed_meta.npz')
x_train = np.load('wsj0_phonelabels_bracketed_train.npy',mmap_mode='r')
y_train = np.load('wsj0_phonelabels_bracketed_train_labels.npy')
# end = x_train.shape[0] % 2048
# x_train = x_train[:-end]
# y_train = y_train[:-end]
nClasses = 138
print nClasses
print 'transforming labels...'
y_train = to_categorical(y_train, num_classes = nClasses)

print 'loading test data...'
x_test = np.load('wsj0_phonelabels_bracketed_dev.npy',mmap_mode='r')
y_test = np.load('wsj0_phonelabels_bracketed_dev_labels.npy')
# # end = x_test.shape[0] % 2048
# # x_test = x_test[:-end]
# # y_test = y_test[:-end]
# # x_test = x_train
# # y_test = y_train
print 'transforming labels...'
y_test = to_categorical(y_test, num_classes = nClasses)

print 'initializing model...'
# model = mlp1(x_train.shape[1], nClasses,2,2048,BN=True)
# model = load_model('test_CP.h5')
model = DBN_DNN(x_train, nClasses,3,2048)
trainNtest(model,x_train,y_train,x_test,y_test,meta,'dbn-3x2048-sig-adagrad','dbn-3x2048-sig-adagrad')

# model = load_model('mlp1-3x2048-sig-adagrad-BN_CP.h5')
# getPreds(model,'SI_ET_20.NDX','/home/mshah1/wsj/wsj0/feat_ci_mls/','.mfc','/home/mshah1/wsj/wsj0/senscores/','.sen')
# pred = model.predict(x_test[:meta['framePos_Dev'][0] - meta['framePos_Train'][-1]],verbose=1)
# writeSenScores('senScores',pred)
# np.save('pred.npy',pred)