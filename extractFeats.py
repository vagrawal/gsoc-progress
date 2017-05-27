from python_speech_features import fbank
import numpy as np
from scipy.io import wavfile
from scipy.signal import hamming
import os
root = '/home/mshah1/wsj/wsj1/'
fileList = root + 'wsj1.wavlist'
with open(fileList,'r') as f:
	files = f.readlines()
	files = map(lambda x: x.strip(), files)

for f in files:
	if not os.path.exists(root + f + '.mls'):
		print f
		(rate,data) = wavfile.read(root + f)
		(feat,_) = fbank(data,samplerate=rate,winlen=0.025,
							winstep=0.015,nfilt=40,nfft=1024,
							lowfreq=250,winfunc=hamming)
		feat = np.log(feat)
		np.savetxt(root + f + '.mls',feat)
	# break
