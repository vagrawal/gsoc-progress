import resnet
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print (y_train.shape)
net = resnet.build_resnet_152(x_train[0].shape,10)
net.compile(optimizer='sgd',
            loss="mean_squared_error",
            metrics=['accuracy'])
net.fit(x_train,y_train,epochs=10)