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
pos=0
def batch_generator(X, labels,framepointer, batch_size):
    global  uttr_no
    print uttr_no
    global pos
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
        #if not uttr:
        #    uttr_strt = 0
        #else:     
        #    uttr_strt = framepointer[uttr-1]+1
        
        #uttr_end = framepointer[uttr]+1
	
	uttr_strt = pos
	uttr_end = pos+framepointer[uttr]
        pos += framepointer[uttr]
        
        print "strt , end",uttr_strt,uttr_end 
        if uttr == len(framepointer)-1:
        
            seq_labels = labels[uttr_strt+1:uttr_end]
        
            seq_labels=np.append(seq_labels,[0])
        
        else:    
            seq_labels = labels[uttr_strt+1:uttr_end+1]
        seq_labels = [int(x+1) for x in seq_labels] #adding 1 to all labels so I can use 0 as mask
        
        
        seq_labels[0]=0 #first symbol is special symbol    
        seq_labels[len(seq_labels)-1]=0 #last symbol is special symbol
        
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

model = load_model("model_3")

#loading test data
Xtest = np.load("/home/hammad/new_data/XDEV.npy")
Ytest = np.load("/home/hammad/new_data/YDEV.npy")
Testframe = np.load("/home/hammad/correct_YDEV.npy")
#Testframe = [int(x-33332446) for x in Testframe]
#Testframe=np.asarray(Testframe)

print "Xdev",Xtest.shape
print "Ydev",Ytest.shape
print "total utterances",Testframe.shape
predictions=[]
count=0
for i in range(len(Testframe)):
    test,testlabel = batch_generator(Xtest,Ytest,Testframe,1)
    pred = model.predict(test, batch_size=1, verbose=1)
    #print pred.shape
    #print pred[0].shape
    pred = [np.delete(i,0) for i in pred[0]]
    #print "removed 0",np.asarray(pred).shape
   
    newp=[]
    count1 = 0
    for x in pred:
        tmp1=[]
        for j in x:
            tmp = float(j)/sum(x)
            tmp1.append(tmp)
        #print "tmp1.shape",len(tmp1)
        if not count1:
            newp = np.asarray([tmp1])
            count1+=1
        else:
            #print "newp.shape",newp.shape
            newp=np.append(newp,np.array([tmp1]),axis=0)
    #pred = [float(j)/sum(x) for x in pred for j in pred[x]]
    #print "checking ",newp[0].shape,newp[5].shape
    print newp.shape
    #if not i%50:
    #score = model.evaluate(test, testlabel, batch_size=1, verbose=1, sample_weight=None)
    #print score
   
    #print pred.shape
    if not count:
        predictions=newp
        count+=1
    else:
        predictions=np.append(predictions,newp,axis=0)
    print "predictions shape",predictions.shape

print np.array(predictions).shape
np.save("DEV_PRED(2)",predictions)
#print np.array(final).shape
#print predictions
