import numpy as np
from python_speech_features import mfcc
from python_speech_features import fbank
from python_speech_features import logfbank
from scipy.signal import hamming
import soundfile as sf
import os


#data prep
from random import randint
import numpy as np
#import pickle
all_feats=[]
all_labels=[]


#for making log feats 
corpus= '/home/hammadabdullah/wsj/wsj0';
f = open('flac_paths.txt','w')
    
for dir_name,subdir_name,files in os.walk(corpus):
    
    for file in files:
        if file.endswith('.flac'):
            path_2_file = dir_name+'/'+file;
            print (path_2_file)
            f.write(path_2_file+'\n')
            (s,r) =sf.read(path_2_file)
            feat = logfbank(s,r, winlen=0.025, winstep=0.01, nfilt=40, nfft=1024, lowfreq=250, highfreq=None, preemph=0.97);
            #np.savetxt('/media/t-rex/F/wsj/wsj_feats/'+file+'.mls',logfeats)
            
            #feat = np.loadtxt('/media/t-rex/F/wsj/wsj_feats/'+file)
            n_frames  = feat.shape[0]

            #assigning random labels 
            labels = [randint(0,2999) for i in range(n_frames)]
            all_labels+=list(labels)
            all_feats += list(feat)
            
#            print(len(feat),len(labels))
print(len(all_feats),len(all_labels))

for_training = int(len(all_feats)*0.8)
Xtrain = all_feats[:for_training]
ytrain = all_labels[:for_training]
Xtest = all_feats[for_training:]
ytest = all_labels[for_training:]

#save as compressed array
#feats are in gigabytes. so compressing them
np.savez('data',Xtrain=Xtrain,ytrain=ytrain,Xtest=Xtest,ytest=ytest)
print("completed")