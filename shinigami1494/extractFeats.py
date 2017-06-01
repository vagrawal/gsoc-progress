from python_speech_features import fbank
import numpy as np
from scipy.io import wavfile
from scipy.signal import hamming
import os
root = '/home/mshah1/wsj/wsj0/'
fileList = root + 'wsj0.wavlist'
with open(fileList,'r') as f:
	files = f.readlines()
	files = map(lambda x: x.strip(), files)

for f in files:
	if not os.path.exists(root + f + '.mls') or 1==1:
		print f
		(rate,data) = wavfile.read(root + f)
		(feat,_) = fbank(data,samplerate=rate,winlen=0.025,
							winstep=0.01,nfilt=40,nfft=1024,
							lowfreq=250,winfunc=hamming)
		feat = np.log(feat)
		np.savetxt(root + f + '.mls',feat)
	# break
