/******************************************************************************
** si_seqint.c
**
** C functions functions for converting between DNA sequences (containing the
** bases A, T, ** G, and C) and lightweight integer representations for
** compact storage or indexing schemes (e.g., bitarrays).
**
** See seqint.h for more documentation.
******************************************************************************/

#include "si_seqint.h"

/* ~~~~~~~~~~~~~~~~~~~~~ Lookup tables for conversions ~~~~~~~~~~~~~~~~~~~~~ */

const char BIN_TO_BASE_LUT[4] = {'A', 'C', 'G', 'T'};

const char BASE_TO_BIN_LUT[128] = {
0,0,0,0,0,0,0,0, // 7
0,0,0,0,0,0,0,0, // 15
0,0,0,0,0,0,0,0, // 23
0,0,0,0,0,0,0,0, // 31
0,0,0,0,0,0,0,0, // 39
0,0,0,0,0,0,0,0, // 47
0,0,0,0,0,0,0,0, // 55
0,0,0,0,0,0,0,0, // 63
0,0,0,1,0,0,0,2, // 71
0,0,0,0,0,0,0,0, // 79
0,0,0,0,3,0,0,0, // 87
0,0,0,0,0,0,0,0, // 95
0,0,0,1,0,0,0,2, // 103
0,0,0,0,0,0,0,0, // 111
0,0,0,0,3,0,0,0, // 119
0,0,0,0,0,0,0,0, // 127
};


uint64_t si_seq2Int(char* seq, int length) {
    uint64_t seqint = 0;
    int i;
    for (i = 0; i < length; i++) {
        seqint <<= 2;
        seqint |= BASE_TO_BIN_LUT[(unsigned char) *seq++];
    }
    return seqint;
}


int si_int2Seq(uint64_t seqint, char* out_str, int length) {
    if (length == 0)
        return -1;
    out_str[length] = '\0';  // Null-terminate C string
    int endptr = length - 1;
    out_str[endptr] = BIN_TO_BASE_LUT[seqint & 0x03];
    for (endptr -= 1; endptr > -1; endptr--) {
        seqint >>= 2;
        out_str[endptr] = BIN_TO_BASE_LUT[seqint & 0x03];
    }
    return 0;
}

uint64_t si_compSeqInt(uint64_t seqint, int length) {
    // Calculate complement of sequence
    uint64_t seqintc = ~seqint;
    // Mask high-order bits
    uint64_t length_mask = ((uint64_t)1 << (length * 2)) - 1;
    seqintc &= length_mask;
    return seqintc;
}

uint64_t si_revCompSeqInt(uint64_t seqint, int length) {
    int bit_ptr = 0;
    int bit_length = length*2;
    uint64_t seqintrc = 0;
    // Bit-wise reverse complement
    for (bit_ptr = 0; bit_ptr < bit_length + 1; bit_ptr += 2) {
        if (!(seqint & ((uint64_t)1 << (bit_ptr)))) {
            // First bit is 0
            if (!(seqint & ((uint64_t)1 << (bit_ptr + 1)))) {
                // Second bit is 0, base is A in seq, should be T in rc
                seqintrc |= ((uint64_t)1 << (bit_length - bit_ptr - 1));
                seqintrc |= ((uint64_t)1 << (bit_length - bit_ptr - 2));
            }
            else { // Second bit is 1, base is C in seq, should be G in rc
                seqintrc |= ((uint64_t)1 << (bit_length - bit_ptr - 2));
            }
        }
        else {
            // First bit is 1
            if (!(seqint & ((uint64_t)1 << (bit_ptr + 1)))) {
                // Second bit is 0, base is G in seq, should be C in rc
                seqintrc |= ((uint64_t)1 << (bit_length - bit_ptr - 1));
            }
            // If second bit is 1, base is T in seq, should be A in seq
        }
    }
    uint64_t length_mask = ((uint64_t)1 << (bit_length)) - 1;
    seqintrc &= length_mask;
    return seqintrc;
}

uint64_t si_revSeqInt(uint64_t seqint, int length) {
    int bit_ptr = 0;
    int bit_length = length*2;
    uint64_t seqintc = 0;
    // Bit-wise reverse
    for (bit_ptr = 0; bit_ptr < bit_length + 1; bit_ptr += 2) {
        if (!(seqint & ((uint64_t)1 << (bit_ptr)))) {
            // First bit is 0
            if ((seqint & ((uint64_t)1 << (bit_ptr + 1)))) {
                // Second bit is 1, base is C in seq
                seqintc |= ((uint64_t)1 << (bit_length - bit_ptr - 1));
            }
        }
        else {
            // First bit is 1
            if (!(seqint & ((uint64_t)1 << (bit_ptr + 1)))) {
                // Second bit is 0, base is G in seq
                seqintc |= ((uint64_t)1 << (bit_length - bit_ptr - 2));
            } else {
                // Second bit is 1, base is T in seq
                seqintc |= ((uint64_t)1 << (bit_length - bit_ptr - 1));
                seqintc |= ((uint64_t)1 << (bit_length - bit_ptr - 2));
            }

        }
    }
    return seqintc;
}

uint64_t si_addBase(uint64_t seqint, char base) {
    seqint <<= 2;
    seqint |= BASE_TO_BIN_LUT[(unsigned char) base];
    return seqint;
}

uint64_t si_removeBase(uint64_t seqint) {
    seqint >>= 2;
    return seqint;
}

uint64_t si_addToWindow(uint64_t seqint, char base, int length) {
    uint64_t length_mask =  ((uint64_t)1 << (length*2)) - 1;
    seqint <<= 2;
    seqint |= BASE_TO_BIN_LUT[(unsigned char) base];
    seqint &= length_mask;
    return seqint;
}

uint64_t si_seqIntSubstring(uint64_t seqint, int start_idx, int end_idx,
                          int length) {
    int trim_right = (length - end_idx) * 2;
    int new_length = (length - start_idx) * 2;
    if (trim_right < 0 || new_length < 0) {
        return seqint;
    }
    uint64_t length_mask = ((uint64_t)1 << (new_length)) - 1;
    seqint >>= trim_right;
    seqint &= length_mask;

    return seqint;
}
