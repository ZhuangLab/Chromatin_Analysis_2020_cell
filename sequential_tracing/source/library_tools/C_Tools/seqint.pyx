cimport numpy as np
ctypedef np.uint64_t uint64_t

cdef uint64_t BASE_TO_BIN_LUT[128]
BASE_TO_BIN_LUT[:] = [0,0,0,0,0,0,0,0,# 7
    0,0,0,0,0,0,0,0, # 15
    0,0,0,0,0,0,0,0, # 23
    0,0,0,0,0,0,0,0, # 31
    0,0,0,0,0,0,0,0, # 39
    0,0,0,0,0,0,0,0, # 47
    0,0,0,0,0,0,0,0, # 55
    0,0,0,0,0,0,0,0, # 63
    0,0,0,1,0,0,0,2, # 71
    0,0,0,0,0,0,0,0, # 79
    0,0,0,0,3,0,0,0, # 87
    0,0,0,0,0,0,0,0, # 95
    0,0,0,1,0,0,0,2, # 103
    0,0,0,0,0,0,0,0, # 111
    0,0,0,0,3,0,0,0, # 119
    0,0,0,0,0,0,0,0]
cdef uint64_t BASE_TO_BIN_LUT_RC[128]
BASE_TO_BIN_LUT_RC[:] = [0,0,0,0,0,0,0,0,# 7
    0,0,0,0,0,0,0,0, # 15
    0,0,0,0,0,0,0,0, # 23
    0,0,0,0,0,0,0,0, # 31
    0,0,0,0,0,0,0,0, # 39
    0,0,0,0,0,0,0,0, # 47
    0,0,0,0,0,0,0,0, # 55
    0,0,0,0,0,0,0,0, # 63
    0,3,0,2,0,0,0,1, # 71
    0,0,0,0,0,0,0,0, # 79
    0,0,0,0,0,0,0,0, # 87
    0,0,0,0,0,0,0,0, # 95
    0,3,0,2,0,0,0,1, # 103
    0,0,0,0,0,0,0,0, # 111
    0,0,0,0,0,0,0,0, # 119
    0,0,0,0,0,0,0,0]


def seq2Int(char *seq):
    cdef int length = len(seq)
    cdef uint64_t seqint = 0
    cdef int i
    
    for i in range(length):
        seqint <<= 2
        seqint|=BASE_TO_BIN_LUT[seq[i]]
    return seqint
def seq2Int_rc(char *seq):
    cdef int length = len(seq)
    cdef uint64_t seqint = 0
    cdef int i
    
    for i in range(length):
        seqint <<= 2
        seqint|=BASE_TO_BIN_LUT_RC[seq[length-i-1]]
    return seqint