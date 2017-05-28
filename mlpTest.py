from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

data = np.load('wsj0_randlabels.npz')
# model = Sequential()
# model.add(Dense(1000, activation='sigmoid', input_dim=40))
# model.add(Dense(1000, activation='sigmoid'))
# model.add(Dense(3000, activation='softmax'))
# model.compile(optimizer='sgd',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# x_train = data['X_Train'][:1000000]
# y_train = data['Y_Train'][:1000000]
# y_train = to_categorical(y_train, num_classes = 3000)
# model.fit(x_train,y_train,epochs=10,batch_size=20000)
# model.save("mlp_randomAlign.h5")

model = load_model("mlp_randomAlign.h5")
x_test = data['X_Test'][:100000]
y_test = data['Y_Test'][:100000]
y_test = to_categorical(y_test, num_classes = 3000)
score = model.evaluate(x_test, y_test, batch_size=2000)
print score