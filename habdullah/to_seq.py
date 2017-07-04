import sys
def to_seq(path_to_oldxtrain,path_to_trainframe,cutoff=500,verbose=0):
    vrb=verbose
    #converting dataset into sequences
    #old(27027263,78)
    #new(46757,max_seq_length,78)
    #will save .npy in current dir
    #print "starting..."
    print "Starting"
    import numpy as np
    newXTrain=[]
    oldXTrain=np.load(path_to_oldxtrain)#"/home/hammad/xtrain.npy"
    oldXFrame=np.load(path_to_trainframe)#"/home/hammad/trainframe.npy"
    uttr=0
    tmp_seq=[]
    flg1=0
    flg2=0
    max_seq=cutoff
    print "old xtrain shape",oldXTrain.shape
    print "frame end pointer shape",oldXFrame.shape
    for i in range(len(oldXFrame)):#0->46k
        if vrb:
            print "uttr no:",i

        if oldXFrame[i]-uttr <max_seq:#zeropad
                if vrb:
                    print "Flag 1 "
                    
                flg1=1
        if oldXFrame[i]-uttr >max_seq:#truncate
                if vrb:
                    print "Flag 2 "
                
                flg2=1
                
        if vrb:        
            print "uttr_start,uttr_end",uttr,oldXFrame[i]

        tmp_seq.append(oldXTrain[uttr:oldXFrame[i],0:78])
        if vrb:
            print "tmp dim:",np.array(tmp_seq).shape
        
        seq_length = oldXFrame[i]-uttr
        uttr = oldXFrame[i]+1  


        if flg1:#zeropad
            if vrb:
                print "max_seq-seq_length",max_seq,seq_length
            
            pad_length = max_seq - seq_length
            pad_matrix=[[[0 for x in range(78)] for y in range(pad_length)]]
            if vrb:
                print "shape of pad matrix",np.array(pad_matrix).shape
            
            tmp_seq= np.append(tmp_seq,pad_matrix,axis=1)
            flg1=0
            tmp_seq=list(tmp_seq)

        if flg2:#truncate
            if vrb:
                print "bef trunc shape",np.array(tmp_seq).shape
                
            tmp = tmp_seq[0]
            tmp_seq=[tmp[0:500,0:79]]
            flg2=0
        if vrb:
            print "tmp dim after modification:",np.array(tmp_seq).shape

        if i==0:
            newXTrain=tmp_seq
            #tmp_seq.dtype
        else:
            newXTrain= np.append(newXTrain,tmp_seq,axis=0)
            if vrb:
                print newXTrain.shape
            newXTrain.astype(np.float32)
        if vrb:
            print "    NEWXTRAIN:",np.array(newXTrain).shape
        if i==5000:
            print "Final shape",newXTrain.shape
            np.save("newXTrain",newXTrain)
            print "saved"
            sys.exit()
        tmp_seq = []
    
    newXTrain.astype(np.float32)
    print "Final shape",newXTrain.shape
    np.save("newXTrain",newXTrain)
    
to_seq("/home/hammad/xtrain.npy","/home/hammad/trainframe.npy",500,1)
