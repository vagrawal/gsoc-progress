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
from keras.models import load_model
#np.set_printoptions(threshold='nan')


def get_session(gpu_fraction=0.1):
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
    uttr_strt = framepointer[uttr-1]+1
    uttr_end = framepointer[uttr]
    return uttr_end - uttr_strt

def batch_generator(X, labels,framepointer, batch_size):
    global  max_len
    uttrences = [np.random.choice(len(framepointer),1,replace=True) for x in range(batch_size)]
    uttr_lengths = [get_uttr_len(uttr,framepointer) for uttr in uttrences]
    max_uttr_len = max(uttr_lengths)[0]
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
            uttr_strt=uttr_strt[0]
        uttr_end = framepointer[uttr]

        uttr_end=uttr_end[0]
        #print uttr
        #print uttr_strt,uttr_end       
        seq_labels = labels[uttr_strt+1:uttr_end+1]
        seq_labels = [int(x+1) for x in seq_labels] #adding 1 to all labels so I can use 0 as mask
        #print "max sequence",max(seq_labels)
        #print len(seq_labels)

        seq_labels[len(seq_labels)-1]=133 #last symbol is special symbol
        #print seq_labels
        uttr_list.append(X[uttr_strt:uttr_end,0:78])
        uttr_list_labls.append(seq_labels)
    print uttrences
    print uttr_lengths    
    print "Inside func,uttr_list",uttr_list_labls
    uttr_list= sequence.pad_sequences(uttr_list, maxlen=max_len, dtype='float32',padding='post',truncating='post' )
    uttr_list_labls = sequence.pad_sequences(uttr_list_labls, maxlen=max_len,dtype='int16',padding='post',truncating='post')
    #print "uttr_list shape",np.array(uttr_list).shape
    #print "uttr_list_labls shape",np.array(uttr_list_labls).shape
    

    #print uttr_list_labls[0],np.array(uttr_list_labls[0]).shape
    new_labels = []
    for label in uttr_list_labls:
            label= to_categorical(label,num_classes=134)
            new_labels.append(label)
    
    #print np.array(uttr_list).shape,np.array(new_labels).shape
    x = np.array(uttr_list)
    y = np.array(new_labels)
    #print "Actual label:",y
    return x.astype(np.float32),y.astype(np.float32)      

model = load_model("newmodel")

#loading test data
Xtest = np.load("/home/hammad/xtest.npy")
Ytest = np.load("/home/hammad/Ytest.npy")
Testframe = np.load("/home/hammad/testframe.npy")
Testframe = [int(x-33332446) for x in Testframe]
Testframe=np.asarray(Testframe)

print "Xtest",Xtest.shape
print "Ytest",Ytest.shape
print "Testframe",Testframe.shape

test,testlabel = batch_generator(Xtest,Ytest,Testframe,1)
pred = model.predict(test, batch_size=1, verbose=1)


print "test",test,test.shape
print "test label",[np.argmax(i) for i in testlabel[0]],testlabel.shape
print "test pred",[np.argmax(i) for i in pred[0]],pred.shape
score = model.evaluate(test, testlabel, batch_size=1, verbose=1, sample_weight=None)
print score
