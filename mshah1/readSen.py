import os
import struct
import numpy as np
from sklearn.linear_model import LinearRegression
def readSen(fname, print_most_prob_sen=False):
	print fname
	f = open(fname,'rb')
	s = ''
	while 'endhdr\n' not in s:
		v = f.read(1)
		s += struct.unpack('s',v)[0]
	magic_num = struct.unpack('I',f.read(4))[0]
	assert magic_num == 0x11223344
	count = 0
	data = []
	while v:
		v = f.read(2)
		if not v:
			continue
		n_active = struct.unpack('h',v)[0]
		# print n_active
		assert n_active == 138

		v = f.read(2*n_active)
		scores = list(struct.unpack('%sh' % n_active, v))
		print np.argmax(scores)
		count += 1
		data += scores
	return np.array(data)

# readSen('../wsj/wsj0/senscores/11_14_1/wsj0/si_et_20/440/440c0401.wv1.flac.sen')
ndx_list = map(lambda x: '../wsj/wsj0/senscores/'+x+'.sen', np.loadtxt('SI_ET_20.NDX',dtype=str))
file_list = map(lambda x: '../wsj/wsj0/sendump_ci/' + x, os.listdir('../wsj/wsj0/sendump_ci/'))
file_list.sort()
file_list = file_list[:-1]
ndx_list = ['../wsj/wsj0/senscores/11_14_1/wsj0/si_et_20/445/445c0403.wv1.flac.sen']
file_list = ['../wsj/wsj0/single_dev/11_14_1/wsj0/si_et_20/445/445c0403.wv1.flac.sen']
x = []
y = []
for i in range(len(file_list)):
	if i < 102:
		print i,ndx_list[i], file_list[i]
		y += list(readSen(ndx_list[i]))
		x += list(readSen(file_list[i]))
		print len(x),len(y), len(x)/138, len(y)/138
		assert len(x) == len(y)
	else:
		print i,ndx_list[i+1], file_list[i]
		y += list(readSen(ndx_list[i+1]))
		x += list(readSen(file_list[i]))
x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)
print x.shape, y.shape
data = np.concatenate((x,y),axis=1)
np.save('data4regression.npy',data)

data = np.load('data4regression.npy')
lreg = LinearRegression(normalize=True,n_jobs=-1)
lreg.fit(data[:,[1]],data[:,[0]])
print lreg.coef_, lreg.intercept_
