import numpy as np

def fileToLabels(data, nPhones=30):	
	nFrames = data.shape[0]
	nPhones = nPhones + np.random.randint(-5,high=5)
	segLen = nFrames / nPhones
	labels = []
	for i in range(nPhones):
		currSegLen = segLen + np.random.randint(-10,high=10)
		labels += [np.random.randint(0,high=3000)]*min(currSegLen,nFrames - len(labels))
	return labels
root = '/home/mshah1/wsj/wsj0/'
fileList = root + 'wsj0.mlslist'
with open(fileList,'r') as f:
	files = f.readlines()
	files = map(lambda x: root + x.strip(), files)
allData = []
allLabels = []
for f in files:
	print f
	data = np.loadtxt(f)
	labels = fileToLabels(data)
	allData += list(data)
	allLabels += list(labels)
lenTrain = len(allData) * 3 / 4
X_Train = allData[:lenTrain]
Y_Train = allLabels[:lenTrain]
X_Test = allData[lenTrain:]
Y_Test = allLabels[lenTrain:]
np.savez('wsj0_randlabels',X_Train=X_Train,Y_Train=Y_Train,X_Test=X_Test,Y_Test=Y_Test)
# sets = np.load('wsj0_randlabels.npz')
# print sets['X_Train'].shape,sets['Y_Train'].shape

