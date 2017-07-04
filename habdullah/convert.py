import numpy as np
import sys

#function that produces deltas and double deltas

file = sys.argv[1]

def deltas(array):
    #check dimensions per frame
    dim = array.shape[1]
    num_frames = array.shape[0]
    #print "dim",dim
    #print "num_frames",num_frames
    delta=[]
    delta_rows=[]
    for i in range (num_frames):
        for j in range(dim):
            t1 = j-1
            t2 = j+1
            if t1 < 0:
                t1 = 0
            if t2 > dim-1:
                t2 = dim-1
        
            delta_tmp = array[i,t2]-array[i,t1]
            delta_rows.append(delta_tmp)
        delta.append(delta_rows)
        delta_rows= []
    delta = np.array(delta)
    return delta

#function to fuse delta and Acc into dataset
def fuse(dataset,delta_array,acc_array):
    new_dataset = []
    dim = dataset.shape[1]
    num_frames = dataset.shape[0]
    for i in range(num_frames):
        a = dataset[i]
        b = delta_array[i]
        c = acc_array[i]
        abc = np.concatenate((a,b,c),axis=0)
        new_dataset.append(abc)
    return np.array(new_dataset)
        
xtrain = np.load(file)
print "Data loaded"
print "before",xtrain.shape

delta = deltas(xtrain)
acc = deltas(delta)
xtrain = fuse(xtrain,delta,acc)

print "after",xtrain.shape
xtrain=xtrain.astype(np.float32)
print "size",xtrain.nbytes
np.save(sys.argv[2],xtrain)
