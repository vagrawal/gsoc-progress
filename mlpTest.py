from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import resnet
import numpy as np

data = np.load('wsj0_randlabels.npz')
# model = Sequential()
# model.add(Dense(1000, activation='sigmoid', input_dim=40))
# model.add(Dense(1000, activation='sigmoid'))
# model.add(Dense(1000, activation='sigmoid'))
# model.add(Dense(1000, activation='sigmoid'))
# model.add(Dense(3000,activation='softmax'))
builder = resnet.ResnetBuilder()
model = builder.build_resnet_152((1,1,40),3000)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
x_train = data['X_Train'][:1000000]
y_train = data['Y_Train'][:1000000]
y_train = to_categorical(y_train, num_classes = 3000)
model.fit(x_train,y_train,epochs=10,batch_size=20000)
model.save("mlp_randomAlign.h5")
