from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Activation, Dropout, add, Conv1D, MaxPooling1D, Reshape, Flatten
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot
def mlp1(input_dim,output_dim,depth,width):
	model = Sequential()
	model.add(Dense(width, activation='relu', input_dim=input_dim))
	for i in range(depth):
		model.add(Dense(width, activation='relu'))
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
	x = Reshape((input_dim,1))(inp)
	x = Conv1D(64,4,padding='same',activation='relu')(x)
	x = MaxPooling1D(3,padding='same')(x)
	x = Flatten()(x)
	x = Dense(width, activation='relu')(x)
	for i in range(nBlocks):
		y = make_dense_res_block(x,2,width)
		x = add([x,y])
	z = Dense(output_dim, activation='softmax')(x)
	model = Model(inputs=inp, outputs=z)
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def preTrain(model,x_train,y_train,skip_layers):
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
		for j in range(i):
			model_new.layers[j].trainable=False
		model_new.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
		print model_new.summary()
		model_new.fit(x_train,y_train,epochs=1,batch_size=2048)
		model.layers[i].set_weights(model_new.layers[-2].get_weights())
	return model

# model = mlp4(20, 132,2,2048)
# print model.summary()
# plot_model(model, to_file='mlp4_model.png')
print 'loading data...'
data = np.load('wsj0_phonelabels.npz')
x_train = data['X_Train']
y_train = data['Y_Train']
print 'transforming labels...'
nClasses = int(np.max(y_train) + 1)
y_train = to_categorical(y_train, num_classes = nClasses)

print 'initializing model...'
model = mlp4(x_train.shape[1], nClasses,1,2048)
print model.summary()
print 'pretraining model...'
model = preTrain(model,x_train[:3000000],y_train[:3000000],range(4))
weights = map(lambda x: x.get_weights(), model.layers)
print 'starting fit...'
history = model.fit(x_train,y_train,epochs=10,batch_size=20000)

print 'saving model...'
model.save('mlp4_phonelabels.h5')
print(history.history.keys())

print 'plotting graphs...'
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('mlp4__relu_adagrad.png')


# model = load_model("mlp1_phonelabels.h5")
print 'loading test data...'
x_test = data['X_Test']
y_test = data['Y_Test']
print 'transforming labels...'
y_test = to_categorical(y_test, num_classes = nClasses)
print 'scoring...'
score = model.evaluate(x_test, y_test, batch_size=2000)
print score
