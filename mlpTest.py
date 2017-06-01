from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import matplotlib as mpl
# print mpl.matplotlib_fname()
# mpl.rcParams['backend'] = 'agg'
# plt = mpl.pyplot
def mlp1():
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=40))
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(3000, activation='softmax'))
	model.compile(optimizer='sgd',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model

def mlp2():
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=40))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3000, activation='softmax'))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='categorical_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	return model

def mlp3():
	model = Sequential()
	model.add(Dense(500, activation='relu', input_dim=40))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(3000, activation='softmax'))
	model.compile(optimizer='adagrad',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model
data = np.load('wsj0_randlabels.npz')
model = mlp1()
# plot_model(model, to_file='mlp3_model.png')
x_train = data['X_Train'][:1000000]
y_train = data['Y_Train'][:1000000]
y_train = to_categorical(y_train, num_classes = 3000)
history = model.fit(x_train,y_train,epochs=10,batch_size=20000)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('mlp1_relu_acc.png')


model = load_model("mlp_randomAlign.h5")
x_test = data['X_Test'][:250000]
y_test = data['Y_Test'][:250000]
y_test = to_categorical(y_test, num_classes = 3000)
with tf.device('/gpu:0'):
	score = model.evaluate(x_test, y_test, batch_size=2000)
print score
