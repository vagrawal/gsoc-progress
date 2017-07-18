import os
CUDA_VISIBLE_DEVICES = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
from keras.models import load_model
import numpy as np
import socket
import struct
import time
def predictFrame(model_name,frame,weight=1,offset=0):
	model = load_model(model_name)
	scores = model.predict(frame)
	
	n_active = scores.shape[1]
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
	print scores
	r_str = struct.pack('%sh' % len(scores[0]), *scores[0])

	# scores /= np.sum(scores,axis=0)
	return r_str

if __name__ == '__main__':
	HOST, PORT = '', 9000
	listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	listen_socket.bind((HOST, PORT))
	listen_socket.listen(1)
	print 'Serving HTTP on port %s ...' % PORT
	while True:
	    client_connection, client_address = listen_socket.accept()
	    packet_len = struct.unpack('i',client_connection.recv(4))[0]
	    print packet_len
	    full_req = ""
	    while len(full_req) < packet_len:
	    	partial_req = client_connection.recv(1024)
	    	full_req += partial_req
	    print len(full_req)
	    [model_name,frame] = full_req.split('\r\n')
	    print model_name
	    frame = list(struct.unpack('%sd' % 440, frame))
	    frame = np.array([frame])
	    resp = str(predictFrame(model_name,frame))
	    print time.time()
	    client_connection.send(resp)
	    print time.time()