from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot
def mlp1(input_dim,output_dim,depth):
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=input_dim))
	for i in range(depth):
		model.add(Dense(1000, activation='relu'))
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
	model.add(Dense(500, activation='relu', input_dim=input_dim))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(output_dim, activation='softmax'))
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def make_dense_res_block(inp, size):
	x = inp
	for i in range(size):
		x = Dense(500, activation='relu')(x)
		x = Dropout(0.25)(x)
	return x

def mlp4(input_dim,output_dim,nBlocks):
	inp = Input(shape=(input_dim,))
	x = Dense(500, activation='relu')(inp)
	for i in range(nBlocks):
		y = make_dense_res_block(x,2)
		x = add([x,y])
	z = Dense(output_dim, activation='softmax')(x)
	model = Model(inputs=inp, outputs=z)
	model.compile(optimizer='rmsprop',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model



# model = mlp4(20, 132,10)
# plot_model(model, to_file='mlp4_model.png')
print 'loading data...'
data = np.load('wsj0_phonelabels.npz')
x_train = data['X_Train']
y_train = data['Y_Train']
print 'transforming labels...'
nClasses = int(np.max(y_train) + 1)
y_train = to_categorical(y_train, num_classes = nClasses)

print 'initializing model...'
model = mlp4(x_train.shape[1], nClasses, 10)
print 'starting fit...'
history = model.fit(x_train,y_train,epochs=10,batch_size=20000)
model.save('mlp4_phonelabels.h5')
print(history.history.keys())

print 'plotting graphs...'
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('mlp4_phonedata_relu_rmsprop.png')


# model = load_model("mlp_randomAlign.h5")
print 'loading test data...'
x_test = data['X_Test']
y_test = data['Y_Test']
y_train = data['Y_Train'].astype(int)
print 'transforming labels...'
y_test = to_categorical(y_test, num_classes = np.max(y_train) + 1)
print 'scoring...'
score = model.evaluate(x_test, y_test, batch_size=2000)
print score
