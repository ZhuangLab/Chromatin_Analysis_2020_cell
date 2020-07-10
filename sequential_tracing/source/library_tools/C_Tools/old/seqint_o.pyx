from cpython.mem cimport (
    PyMem_Malloc,
    PyMem_Realloc,
    PyMem_Free
)
cimport numpy as np
ctypedef np.uint64_t uint64_t
from typing import Union

STR_T = Union[str, bytes]

cdef extern from "si_seqint.h":
    uint64_t si_seq2Int(char*, int)
    int si_int2Seq(uint64_t, char*, int)
    uint64_t si_revSeqInt(uint64_t, int)
    uint64_t si_compSeqInt(uint64_t, int)
    uint64_t si_revCompSeqInt(uint64_t, int)
    uint64_t si_addBase(uint64_t, char)
    uint64_t si_removeBase(uint64_t)
    uint64_t si_seqIntSubstring(uint64_t, int, int, int)
    uint64_t si_addToWindow(uint64_t, char, int)

    
#######################cutil    
from cpython.ref cimport PyObject
from cpython.mem cimport (
    PyMem_Malloc,
    PyMem_Realloc,
    PyMem_Free
)
from libc.string cimport (
    memset,
    memcpy
)
from libc.stdlib cimport (
    malloc,
    free
)

cdef inline bytes _bytes(s):
    if isinstance(s, str):
        # encode to the specific encoding used inside of the module
        return (<str>s).encode('utf8')
    else:
        return s
# end def

cdef extern from "Python.h":
    cdef bint PyBytes_Check(object)
    cdef int PyBytes_AsStringAndSize(object, char **, Py_ssize_t *)
    cdef object PyBytes_FromString(const char*)
    cdef object PyBytes_FromStringAndSize(const char *, Py_ssize_t)
    cdef char* PyBytes_AsString(object)

    cdef char* PyUnicode_AsUTF8AndSize(object, Py_ssize_t *)
    cdef object PyUnicode_FromString(const char *)
    cdef object PyUnicode_FromStringAndSize(const char *, Py_ssize_t)
    cdef char* PyUnicode_AsUTF8(object)    
    
"""
The following functions take a
o1 - python string object
length - a pointer that will store the length of the string
returns:
    c_str1: string pointer to the internal string of the o1
For Python 3 it can take a bytes object or a unicode object
"""
cdef inline char* obj_to_cstr(object o1):
    cdef char* c_str1
    cdef Py_ssize_t length
    if PyBytes_Check(o1):
        if PyBytes_AsStringAndSize(o1, &(c_str1), &length) == -1:
            raise TypeError("obj_to_cstr:")
        return c_str1
    else:
        c_str1 = PyUnicode_AsUTF8AndSize(o1, &length)
        if c_str1 == NULL:
            raise OSError("obj_to_cstr:")
    return c_str1
# end def    
####################################   cutil
"""Compact integer representations of DNA sequences\n
"""

def seq2Int(seq_obj: STR_T) -> float:
    """Convert a DNA sequence to an integer representation\n\n
    seq: DNA sequence comprised of the bases A/a, T/t, G/g, and C/c\n
    """
    cdef char*           seq
    cdef uint64_t        seqint
    cdef int length

    length = <int> len(seq_obj)

    if length > 30:
        raise ValueError("seq2Int: Sequences over 30 bp cannot be converted (%d)" % length)

    seq = obj_to_cstr(seq_obj)

    seqint = si_seq2Int(seq, length)
    return seqint
# end def

def reverseComplement(uint64_t seqint, int length) -> float:
    """Return the reverse complement seqint representation of `seqint`\n\n
    seqint: integer representation of a DNA sequence\n
    length: length of the represented sequence in bases\n
    """
    cdef uint64_t         seqintrc

    seqintrc = si_revCompSeqInt(seqint, length)

    return seqintrc
# end def


def complement(uint64_t seqint, int length) -> float:
    """Return the complement seqint representation of `seqint`\n\n
    seqint: integer representation of a DNA sequence\n
    length: length of the represented sequence in bases\n
    """
    cdef uint64_t         seqintrc

    seqintrc = si_compSeqInt(seqint, length)

    return seqintrc
# end def

def reverse(uint64_t seqint, int length) -> float:
    """Return the reverse complement seqint representation of `seqint`\n\n
    seqint: integer representation of a DNA sequence\n
    length: length of the represented sequence in bases\n
    """
    cdef uint64_t         seqintrc

    seqintrc = si_revSeqInt(seqint, length)

    return seqintrc
# end def


def addBase(uint64_t seqint, base_obj: STR_T) -> float:
    """Add a base to the right-hand side of a seqint\n\n
    seqint: integer representation of a DNA sequence\n
    base: base (A/a, T/t, G/g, or C/c) to be added\n
    length: length of the represented sequence in bases\n
    """
    cdef char*       base
    cdef  uint64_t    seqintmod

    cdef bytes base_py = _bytes(base_obj)
    base = base_py

    seqintmod = si_addBase(seqint, base[0])

    return seqintmod
# end def


def removeBase(uint64_t seqint, int length) -> float:
    """Remove a base from the right-hand side of a seqint\n\n
    seqint: integer representation of a DNA sequence\n
    length: length of the represented sequence in bases\n
    """
    cdef uint64_t    seqintmod

    seqintmod = si_removeBase(seqint)

    return seqintmod
# end def

def addToWindow(uint64_t seqint, base_obj: STR_T, int length) -> float:
    """Add a base to the right-hand side of a seqint (fixed length window)\n\n
    seqint: integer representation of a DNA sequence\n
    base: base (A/a, T/t, G/g, or C/c) to be added\n
    length: length of the represented sequence in bases\n
    """
    cdef char*       base
    cdef uint64_t    seqintmod

    cdef bytes base_py = _bytes(base_obj)
    base = base_py

    seqintmod = si_addToWindow(seqint, base[0], length)

    return seqintmod
# end def

def getSubstring(uint64_t seqint, int start_idx, int end_idx, int length) -> float:
    """Return the seqint representing the defined substring

    Args:
        seqint: integer representation of a DNA sequence
        start_idx: substring start index (inclusive, 0-based)\n
        start_idx: substring end index (exclusive)\n
        length: length of the represented sequence in bases\n

    Returns:
        float
    """
    cdef uint64_t    seqintmod

    seqintmod = si_seqIntSubstring(seqint, start_idx, end_idx, length);

    return seqintmod
# end def

def int2Seq(uint64_t seqint, int length) -> str:
    """Return the DNA sequence string represented by ``seqint``

    Args:
        seqint: integer representation of a DNA sequence
        length: length of the represented sequence in bases
    """
    cdef object ret_obj
    cdef char* out_seq

    if length == 0:
        return ""

    out_seq = <char *> PyMem_Malloc((length+1)*sizeof(char))
    if out_seq == NULL:
        raise OSError("Could not allocate memory for sequence.")

    if si_int2Seq(seqint, out_seq, length) != 0:
        PyMem_Free(out_seq)
        raise OSError("Could not convert integer to sequence.")

    try:
        ret_obj = out_seq.decode('UTF-8')
    finally:
        PyMem_Free(out_seq)

    return ret_obj
# end def

cdef inline int2Seq_c(uint64_t seqint, char* out, int length):
    """cython header exposed wrapper for si_int2Seq
    """
    return si_int2Seq(seqint, out, length)
