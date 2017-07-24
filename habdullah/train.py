#importing required libraries
from keras import optimizers
from keras.models import Sequential
from keras.layers import TimeDistributed,Dense, Dropout, Activation,LSTM,Bidirectional,GaussianNoise
from keras.optimizers import SGD
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing import sequence
#supressing warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


def get_session(gpu_fraction=0.95):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
max_len = 0


def get_uttr_len(uttr,framepointer):
    if not uttr:
        uttr_strt = 0
    else:     
        uttr_strt = framepointer[uttr-1]+1
            #uttr_strt=uttr_strt[0]
    uttr_end = framepointer[uttr]
    return int(uttr_end - uttr_strt)

def batch_generator(X, labels,framepointer, batch_size):
    while True: 
        global  max_len
        
        uttrences = [int(np.random.choice(len(framepointer)-1,1,replace=True)) for x in range(batch_size)]
        uttr_lengths = [get_uttr_len(uttr,framepointer) for uttr in uttrences]
        max_uttr_len = max(uttr_lengths)
        max_len = max_uttr_len
        #print "max uttr len",max_uttr_len
        #builds a batch of size N=len(uttrences) 

        uttr_list = []
        uttr_list_labls=[]
        for uttr in uttrences:
            if not uttr:
                uttr_strt = 0
            else:     
                uttr_strt = framepointer[uttr-1]+1


            uttr_end = framepointer[uttr]        

            seq = X[uttr_strt:uttr_end,0:75]
            seq_labels = labels[uttr_strt+1:uttr_end+1]
            #adding 1 to all labels so I can use 0 as mask
            #seq_labels = [int(x+1) for x in seq_labels] 
            #seq_labels[len(seq_labels)-1]= #last symbol is special symbol
            uttr_list.append(seq)
            uttr_list_labls.append(seq_labels)
        uttr_list= sequence.pad_sequences(uttr_list, maxlen=max_len, dtype='float32',padding='post',truncating='post',value=0.)
        uttr_list_labls = sequence.pad_sequences(uttr_list_labls,maxlen=max_len,\
                          dtype='int16',padding='post',truncating='post',value=138.)
       
        new_labels = []
        for label in uttr_list_labls:
            label= to_categorical(label,num_classes=139)
            new_labels.append(label)
            #print np.array(uttr_list).shape,np.array(new_labels).shape
        x = np.array(uttr_list)
        y = np.array(new_labels)
        yield x.astype(np.float32), y.astype(np.float32)
      
        
print "loading data"
#loading data from compressed arrays
import numpy as np
X = np.load("/home/hammad/new_data/XTRAIN.npy")
Y = np.load("/home/hammad/new_data/YTRAIN.npy")
frame = np.load("/home/hammad/new_data/YTRAIN(1).npy")

#meta= np.load('wsj0_phonelabels_meta.npz')
#xdev = np.load('wsj0_phonelabels_dev.npy')
#ytest =  meta['Y_Test']

print "Data loaded"
print "X",X.shape
print "Y",Y.shape
print "frame (total utterances)",frame.shape

#window size
cutoff =500

from keras.utils import plot_model

#model = Sequential()
#model.add(Bidirectional(LSTM(150,return_sequences=True), input_shape=(cutoff, 78)))
#model.add(Dropout(0.3))
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#lstm_hist=model.fit(Xtrain, Ytrain, epochs=50, validation_split=0.3,batch_size=400, verbose=2)
#lstm_hist=model.fit_generator(batch_generator(X,Y,frame,500,1),steps_per_epoch=50,epochs=25,verbose=2)


model = Sequential()
#5 bidirectional hidden layers
model.add(Bidirectional(LSTM(150,return_sequences= True) ,input_shape=(None, 75)))
model.add(GaussianNoise(0.075))
model.add(Bidirectional(LSTM(150,return_sequences= True) ,input_shape=(None, 75)))
model.add(TimeDistributed(Dense(139)))
model.add(Activation('softmax'))

print model.summary()

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

lstm_hist=model.fit_generator(batch_generator(X,Y,frame,38),steps_per_epoch=467,epochs=7,verbose=2)
model.save("model_2")

X =0
Y=0


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print "plotting"
plt.plot(lstm_hist.history['acc'])
#plt.plot(lstm_hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='lower right')
#plt.show()
plt.savefig('acc.png')

plt.plot(lstm_hist.history['loss'])
#plt.plot(lstm_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='lower right')
#plt.show()
plt.savefig('loss.png')


#loading test data
Xtest = np.load("/home/hammad/new_data/XDEV.npy")
Ytest = np.load("/home/hammad/new_data/YDEV.npy")
Testframe = np.load("/home/hammad/new_data/YDEV(1).npy")
#Testframe = [int(x-33332446) for x in Testframe]
#Testframe=np.asarray(Testframe)

print "XDEV",Xtest.shape
print "YDEV",Ytest.shape
print "frame (total uttr)",Testframe.shape
score = model.evaluate_generator(batch_generator(Xtest,Ytest,Testframe,1),1103)
print score

Xtest = np.load("/home/hammad/new_data/XTEST.npy")
Ytest = np.load("/home/hammad/new_data/YTEST.npy")
Testframe = np.load("/home/hammad/new_data/YTEST(1).npy")
#Testframe = [int(x-33332446) for x in Testframe]
#Testframe=np.asarray(Testframe)

print "Xtest",Xtest.shape
print "Ytest",Ytest.shape
print "frame (total uttr)",Testframe.shape
score = model.evaluate_generator(batch_generator(Xtest,Ytest,Testframe,1),520)
print score