#importing required libraries
from keras import optimizers
from keras.models import Sequential
from keras.layers import *
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.preprocessing import sequence
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard,CSVLogger,Callback,ModelCheckpoint

#from keras.optimizers import SGD
import numpy as np
import os
import fst
#supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from time import time

def get_session(gpu_fraction=0.5):
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

def best_path(x,y,z):
    #make weighted transducer 
    t1= w_transducer(x,y,z)
    t2 = transducer(x,y,z)
    t3 = t>>t2
    t= t3.shortest_path()
    predicted=[]
    for state in t.states:
        for arc in state.arcs:
            predicted.append(t.osyms.find(arc.olabel))
            # print state.stateid,arc.nextstate,\
            # t.isyms.find(arc.ilabel),\
            # t.osyms.find(arc.olabel),arc.weight
    print "Now generating best path seq labels"


def w_transducer(seq,uttrs,seq_labels):
    global tru_labels
    import math
    t = fst.Transducer()
    sym=fst.SymbolTable() 
    #for i in range(138):
    x=0
    
    labels = [x for i in range(139)]
    symbols = [i for i in range(139)]#tru_labels#['G1','G2','G3','UH1','UH2','UH3','D1','D2','D3']
    label_prob = model.predict([seq], batch_size=1, verbose=1)
    #print "this should be 139",len(labels)
    #print labels
    for j in range(len(label_prob)):
        prob_slice = label_prob[j]
        #label_prob = ran_lab_prob()
        #labels=[i for i in range(138)]
        print "this should be 139",len(label_prob[0])
        for i in range(len(label_prob[0])):
            prob =  prob_slice[i] #"%.4f" %
            t.add_arc(0+x, 1+x,str(labels[i]+str(j)),symbols[i],math.log(prob))
        x+=1
    t[len(label_prob)].final = -1
    return t

def transducer(seq,uttr,seq_labels):
    global trans,cmudict,sym2state
    t2=fst.Transducer()
    transcript = trans[uttr]
    labels=[]
    symbols=[]
    for i in transcript:
        labels.append(cmudict[i])
    #labels = [for i in transcript]
    for i in labels:
        symbols.append(sym2state[i])#[i+"1",i+"2",i+"3"]
    print "labels",labels
    print  "symbols",symbols
    print "trans",transcript
    #symbols = ['G1','G2','G3','UH1','UH2','UH3','D1','D2','D3']
    #labels  = ['G','UH','D']
    x=0
    count=0
    for i in range(1,len(symbols)+1):
        if i%3==1:
            t2.add_arc(0+x,1+x,symbols[x],str(labels[count]+"/"+"("+symbols[x]+")"))
        else:
            t2.add_arc(0+x,1+x,symbols[x],str(sym.find(0)+"("+symbols[x]+")"))
        t2.add_arc(1+x,1+x,symbols[x],str(sym.find(0)+"("+symbols[x]+")"))
        print "i",i
        if i%3==0:
            count+=1
        x+=1
        
    t2[len(symbols)].final=True
    return t2

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
        global epoch_no
        
        uttrences = [int(np.random.choice(len(framepointer)-1,1,replace=True)) for x in range(batch_size)]
        #uttrences = [0,1,2,3,4,5]
        uttr_lengths = [get_uttr_len(uttr,framepointer) for uttr in uttrences]
        max_uttr_len = max(uttr_lengths)
        max_len = max_uttr_len
        #print "max uttr len",max_uttr_len
        #builds a batch of size N=len(uttrences) 
        uttr_list = []
        uttr_list_labls=[]
        for uttr in uttrences:
            #uttr =1
            if not uttr:
                uttr_strt = 0
            else:     
                uttr_strt = framepointer[uttr-1]
            uttr_end = framepointer[uttr]
            seq = X[uttr_strt:uttr_end,0:75]
            seq_labels = labels[uttr_strt+1:uttr_end+1]
            if epoch_no <= 2 or epoch_no+1==Num_epochs:
                seq_labels = labels[uttr_strt+1:uttr_end+1]
            else:
                print "epoch is greater than 3.labels now supplied by best path fucntion"
                best_path(seq,uttr,seq_labels)
            seq_labels = [int(x+1) for x in seq_labels] 
            #uttr_end = framepointer[uttr]+1        
            #print "uttr strt,end",uttr_strt,uttr_end
            #print np.array(seq).shape
            #adding 1 to all labels so I can use 0 as mask
            #seq_labels[len(seq_labels)-1]= #last symbol is special symbol
            uttr_list.append(seq)
            uttr_list_labls.append(seq_labels)
        #print "uttr list",uttr_list
        #print "uttr_labels",uttr_list_labls
        uttr_list= sequence.pad_sequences(uttr_list, maxlen=max_len, dtype='float32',padding='post',truncating='post',value=0.)
        uttr_list_labls = sequence.pad_sequences(uttr_list_labls,maxlen=max_len,\
                          dtype='int16',padding='post',truncating='post',value=0.)
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
frame = np.load("/home/hammad/new_data/XTRAIN_FRAMEPOS.npy")
f = np.load("/home/hammad/new_data/YTRAIN(1).npy")
trans = np.load("trans.npy")
cmudict = np.load("cmudict.dict.npy").items()
tru_labels = np.load("tru_labels.npy")
sym2state = np.load("sym2state.npy")
print frame
print f
frame=f
f=0
#meta= np.load('wsj0_phonelabels_meta.npz')
#xdev = np.load('wsj0_phonelabels_dev.npy')
#ytest =  meta['Y_Test']

print "Data loaded"
print "X",X.shape
print "Y",Y.shape
print "frame (total utterances)",frame.shape

#for testing
Xtest = np.load("/home/hammad/new_data/XTEST.npy")
Ytest = np.load("/home/hammad/new_data/YTEST.npy")
Testframe = np.load("/home/hammad/new_data/YTEST(1).npy")

#window size
#cutoff =500
epoch_no=0
Num_epochs=5
class Access_epochnum_callback(Callback):
    def on_epoch_end(self, epoch,logs={}):
        global epoch_no
        #print "\nEpoch number",epoch,"ended\n"
        epoch_no=epoch
checkpoint=ModelCheckpoint("model_4",verbose=1)
access_epoch = Access_epochnum_callback()
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
csvlogger = CSVLogger('train.log')

#model = Sequential()
#model.add(Bidirectional(LSTM(150,return_sequences=True), input_shape=(cutoff, 78)))
#model.add(Dropout(0.3))
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#lstm_hist=model.fit(Xtrain, Ytrain, epochs=50, validation_split=0.3,batch_size=400, verbose=2)
#lstm_hist=model.fit_generator(batch_generator(X,Y,frame,500,1),steps_per_epoch=50,epochs=25,verbose=2)


model = Sequential()
#5 bidirectional hidden layers
model.add(Masking(mask_value=138., input_shape=(None,75)))
model.add(Bidirectional(LSTM(150,return_sequences= True)))
model.add(GaussianNoise(0.075))
#model.add(Bidirectional(LSTM(150,return_sequences= True) ,input_shape=(None, 75)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(139)))
model.add(Activation('softmax'))

print model.summary()
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
rms=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['accuracy'])

lstm_hist=model.fit_generator(batch_generator(X,Y,frame,50),\
        steps_per_epoch=491,epochs=20,verbose=2,\
        validation_data=batch_generator(Xtest,Ytest,Testframe,1),validation_steps=1,\
        callbacks=[tensorboard,access_epoch,checkpoint])
model.save("model_4")

X=0
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
