
uttr_no = 0
def get_uttr_len(uttr,framepointer):
    if not uttr:
        uttr_strt = 0
    else:     
        uttr_strt = framepointer[uttr-1]+1
    uttr_end = framepointer[uttr]
    return int(uttr_end - uttr_strt)

def batch_generator(X,framepointer, batch_size):
    while True: 
        global  max_len
        global epoch_no
        global uttr_no
        uttrences = [uttr_no]
        uttr_no+=1
        #uttrences = [0,1,2,3,4,5] 
        uttr_list = []
        for uttr in uttrences:
            if not uttr:
                uttr_strt = 0
            else:     
                uttr_strt = framepointer[uttr-1]
            print "decoding uttr no",uttr
            uttr_end = framepointer[uttr]
            seq = X[uttr_strt:uttr_end,0:num_filters]
            #seq_labels = labels[uttr_strt+1:uttr_end+1]
            uttr_list.append(seq)
        x = np.array(uttr_list)
        #y = np.array(new_labels)
        return x.astype(np.float32)

print "Loading Data"
#Load the model which will be evaluate each utterance
model = load_model("model_4")

#load data arrays
data = np.load("features.npy")
frames=np.load("frame_pointer.npy")

#load data
x_test=data
frames=framepointer
print "x_train shape",x_test.shape
print "total utterances",frames.shape

predictions=[]
count=0

print "Starting NN evaluation..."
for i in range(len(Testframe)):
    test = batch_generator(x_test,frames,1)
    pred = model.predict(test, batch_size=1, verbose=1)
    if not count:
        predictions=pred[0]
        count+=1
    else:
        #print pred[0].shape
        predictions=np.append(predictions,pred[0],axis=0)
    print "predictions shape",predictions.shape

print "Shape of predictions array",np.array(predictions).shape
print "Saving predictions as preds.npy"
np.save("preds",predictions)

print "NN evaluation finished"
print "Writing Senone files..."


def writeSenScores(filename,scores,freqs,weight,offset):
    n_active = scores.shape[1]
    s = ''
    s = """s3
version 0.1
mdef_file ../../en_us.cd_cont_4000/mdef
n_sen 138
logbase 1.000100
endhdr
"""
    s += struct.pack('I',0x11223344)
    # print freqs
    # scores /= freqs + (1.0 / len(freqs))
    scores = np.log(scores)/np.log(1.0001)
    scores *= -1
    scores -= np.min(scores,axis=1).reshape(-1,1)
    # scores = scores.astype(int)
    scores *= 0.1 * weight
    scores += offset
    truncateToShort = lambda x: 32676 if x > 32767 else (-32768 if x < -32768 else x)
    vf = np.vectorize(truncateToShort)
    scores = vf(scores)
    # scores /= np.sum(scores,axis=0)
    for r in scores:
        r = r[1:]
        # print np.argmin(r)
        s += struct.pack('h',n_active)
        r_str = struct.pack('%sh' % len(r), *r)
        # r_str = reduce(lambda x,y: x+y,r_str)
        s += r_str
    with open(filename,'w') as f:
        f.write(s)

def getPredsFromArray(nFrames,filenames,res_dir,res_ext,freqs=0):
    preds = np.load("dev_pred_pad_138.npy")
    for i in range(len(nFrames)):
        fname = filenames[i]
        #fname = reduce(lambda x,y: x+'/'+y,fname.split('/')[4:])
        #stdout.write("\r%d/%d  " % (i,len(filenames)))
        #stdout.flush()
        #print fname
        res_file_path = res_dir+"/"+fname+res_ext
        #print res_file_path
        dirname = os.path.dirname(res_file_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not i:
            uttr_strt = 0
        else:
            uttr_strt = nFrames[i-1]
        #print "decoding uttr no",i
        uttr_end = nFrames[i]
        print "strt,end=",uttr_strt,uttr_end
        print i,"/",len(filenames)
        writeSenScores(res_file_path,preds[uttr_strt:uttr_end],freqs,1,0)


#model=load_model("newmodel")
#nFrames=np.load("/home/hammad/new_data/YDEV(1).npy")
ctl_file = #-----> CTL FILE
res_dir = #-----> full path to output folder.
filenames = ctl_file
res_dir="/home/hammad/senfiles"
res_ext=".sen"
getPredsFromArray(nFrames,filenames,res_dir,res_ext)
print "finished Writing senone files!"
