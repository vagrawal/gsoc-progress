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


#KTF.set_session(get_session())
uttr_no = 0


def get_uttr_len(uttr,framepointer):
    if not uttr:
        uttr_strt = 0
    else:     
        uttr_strt = framepointer[uttr-1]+1
            #uttr_strt=uttr_strt[0]
    uttr_end = framepointer[uttr]
    return int(uttr_end - uttr_strt)

def batch_generator(X, labels,framepointer, batch_size):
    global  uttr_no
    print uttr_no
    #uttrences = [int(np.random.choice(len(framepointer)-1,1,replace=False)) for x in range(batch_size)]
    #uttr_lengths = [get_uttr_len(uttr,framepointer) for uttr in uttrences]
    uttrs=[uttr_no]
    uttr_no=uttr_no+1
    uttr_lengths = get_uttr_len(uttrs[0],framepointer)
    #max_uttr_len = max(uttr_lengths)
    #max_len = max_uttr_len
    #print "utter",uttrs
    #print "uttr_length",uttr_lengths
    #print "max uttr len",max_uttr_len

    #builds a batch of size N=len(uttrences) 

    uttr_list = []
    uttr_list_labls=[]
    for uttr in uttrs:
        if not uttr:
            uttr_strt = 0
        else:     
            uttr_strt = framepointer[uttr-1]+1
            #uttr_strt=uttr_strt[0]
        uttr_end = framepointer[uttr]

        #uttr_end=uttr_end[0]
        #print uttr
        #print uttr_strt,uttr_end 
        if uttr == len(framepointer)-1:
            #print "len(framepointer)",len(framepointer),"-1,uttr",uttr
            seq_labels = labels[uttr_strt+1:uttr_end]
            #print seq_labels.shape
            seq_labels=np.append(seq_labels,[0])
            #print seq_labels.shape
        else:    
            seq_labels = labels[uttr_strt+1:uttr_end+1]
            seq_labels = [int(x+1) for x in seq_labels] #adding 1 to all labels so I can use 0 as mask
        #print "max sequence",max(seq_labels)
        #print len(seq_labels)
        seq_labels[0]=0 #first symbol is special symbol    
        seq_labels[len(seq_labels)-1]=0 #last symbol is special symbol
        #print seq_labels
        uttr_list.append(X[uttr_strt:uttr_end,0:75])
        uttr_list_labls.append(seq_labels)
       
    #print "Inside func,uttr_list",uttr_list_labls
    #uttr_list= sequence.pad_sequences(uttr_list, maxlen=max_len, dtype='float32',padding='post',truncating='post' )
    #uttr_list_labls = sequence.pad_sequences(uttr_list_labls, maxlen=max_len,dtype='int16',padding='post',truncating='post')
    #print "uttr_list shape",np.array(uttr_list).shape
    #print "uttr_list_labls shape",np.array(uttr_list_labls).shape
    

    #print uttr_list_labls[0],np.array(uttr_list_labls[0]).shape
    new_labels = []
    for label in uttr_list_labls:
            label= to_categorical(label,num_classes=139)
            new_labels.append(label)
    
    #print np.array(uttr_list).shape,np.array(new_labels).shape
    x = np.array(uttr_list)
    y = np.array(new_labels)
    #print "Actual label:",y
    return x.astype(np.float32),y.astype(np.float32)      

model = load_model("model_2")

#loading test data
Xtest = np.load("/home/hammad/new_data/XDEV.npy")
Ytest = np.load("/home/hammad/new_data/YDEV.npy")
Testframe = np.load("/home/hammad/new_data/YDEV(1).npy")
#Testframe = [int(x-33332446) for x in Testframe]
#Testframe=np.asarray(Testframe)

print "Xdev",Xtest.shape
print "Ydev",Ytest.shape
print "total utterances",Testframe.shape
predictions=[]
for i in range(1103):
    test,testlabel = batch_generator(Xtest,Ytest,Testframe,1)
    pred = model.predict(test, batch_size=1, verbose=1)
    #print pred.shape
    
    if not i%50:
        score = model.evaluate(test, testlabel, batch_size=1, verbose=1, sample_weight=None)
        print score
    #testlabel = [np.argmax(i) for i in testlabel[0]]
    #pred =[np.argmax(i) for i in pred[0]]
    
    #print "test",test,test.shape
    #print "test label",testlabel,np.array(testlabel).shape
    #print "test pred",pred,np.array(pred).shape
    #pred=[list(x) for x in pred]
    if not i:
        predictions=pred[0]
    else:
        predictions=np.append(predictions,pred[0],axis=0)
    print "predictions shape",predictions.shape

print np.array(predictions).shape
np.save("DEV_PRED(1)",predictions)
#print np.array(final).shape
#print predictions
