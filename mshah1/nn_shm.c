#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

int main(){
	int packet_len = (440*4);
    int n_feats = 225
    int n_scores = 138
	// int server_sock = open_server_socket("127.0.0.1","0.0.0.0",9000);
    int shm_fd_scr = shm_open("/shm_keras_scores",O_RDWR || O_CREATE)
    ftruncate(shm_fd_scr,(n_scores + 1)*2)
    int16_t *scrs = mmap(NULL,(n_scores + 1) * 2, PROT_READ, MAP_SHARED, shm_fd_scr, 0)
    int shm_fd_feat = shm_open("/shm_keras_feats",O_RDWR || O_CREATE)
    ftruncate(shm_keras_feats,(n_feats + 1)*4)
    float *feat = mmap(NULL,(n_feats + 1) * 4, PROT_WRITE, MAP_SHARED, shm_fd_feat, 0)
    
    memcpy(&feat[1], feat_buf_ci, n_feats*4)
	return 0;
}