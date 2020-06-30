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
c_str2 - a pointer to the c string of the copy of o1
returns:
    o2 - a python string object

For Python 3 it can take a bytes object or a unicode object

This is unsafe for single characters in cpython due to object reuse
https://github.com/python/cpython/blob/master/Objects/unicodeobject.c#L4688
"""
cdef inline object copy_obj_to_cstr_unsafe( object      o1,
                                            Py_ssize_t* length,
                                            char**      c_str2):

    cdef object o2
    cdef char* c_str1
    cdef size_t b_length
    if PyBytes_Check(o1):
        if PyBytes_AsStringAndSize(o1, &(c_str1), length) == -1:
            raise TypeError("copy_obj_to_cstr:")
        b_length = length[0]
        o2 = PyBytes_FromStringAndSize(c_str1, b_length)
        #if o2 == NULL:
        #    raise OSError("copy_obj_to_cstr:")
        c_str2[0] = PyBytes_AsString(o2)
    else:
        c_str1 = PyUnicode_AsUTF8AndSize(o1, length)
        if c_str1 == NULL:
            raise OSError("copy_obj_to_cstr:")
        b_length = length[0]
        o2 = PyUnicode_FromStringAndSize(<const char *>c_str1, b_length)
        #if o2 == NULL:
        #    raise OSError("copy_obj_to_cstr:")
        c_str2[0] = PyUnicode_AsUTF8(o2)

    return o2
# end def

"""
The following functions take a

o1 - python string object
length - a pointer that will store the length of the string
c_str2 - a pointer to the c string of the copy of o1
returns:
    obj_type - type a python string object for Python 3
        1 == bytes
        0 == str (unicode)

For Python 3 it can take a bytes object or a unicode object

"""
cdef inline int copy_obj_to_cstr(   object      o1,
                                    Py_ssize_t* length,
                                    char**      c_str2) except -1:

    cdef char* c_str1 = NULL
    cdef char* temp = NULL
    cdef size_t b_length
    cdef int obj_type = 0
    if PyBytes_Check(o1):
        obj_type = 1
        if PyBytes_AsStringAndSize(o1, &(c_str1), length) == -1:
            #raise TypeError("copy_obj_to_cstr:")
            return -1
    else:
        obj_type = 0
        c_str1 = PyUnicode_AsUTF8AndSize(o1, length)
        if c_str1 == NULL:
            #raise OSError("copy_obj_to_cstr:")
            return -1
    b_length = length[0]+1
    #temp = <char *> PyMem_Malloc(b_length*sizeof(char))
    temp = <char *> malloc(b_length*sizeof(char))
    if temp == NULL:
        return -1
    memcpy(temp, c_str1, b_length)
    c_str2[0] = temp
    return obj_type
# end cdef

"""
The following functions take a

c_str - a pointer to the c string of the copy of o1
length - the length of the string
obj_type - for Python 3
        1 == bytes
        0 == str (unicode)

returns:
    obj - a python string object and frees the memory associated with c_str
"""
cdef inline cstr_to_obj(char*       c_str,
                        Py_ssize_t  length,
                        int         obj_type):
    cdef object obj
    if obj_type:
        obj = PyBytes_FromStringAndSize(c_str, length)
    else:
        obj = PyUnicode_FromStringAndSize(<const char *>c_str, length)
    #PyMem_Free(c_str)
    free(c_str)
    return obj
# end cdef
cdef inline cstr_to_obj_nofree( char*       c_str,
                                Py_ssize_t  length,
                                int         obj_type):
    cdef object obj
    if obj_type:
        obj = PyBytes_FromStringAndSize(c_str, length)
    else:
        obj = PyUnicode_FromStringAndSize(<const char *>c_str, length)
    return obj
# end cdef

cdef inline cstr_to_obj_nolength(   char*   c_str,
                                    int     obj_type):
    cdef object obj
    if obj_type:
        obj = PyBytes_FromString(c_str)
    else:
        obj = PyUnicode_FromString(<const char *>c_str)
    return obj
# end cdef


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

"""
Same as above but fetches the string length too
"""
cdef inline char* obj_to_cstr_len(  object      o1,
                                    Py_ssize_t* length):
    cdef char* c_str1
    if PyBytes_Check(o1):
        if PyBytes_AsStringAndSize(o1, &(c_str1), length) == -1:
            raise TypeError("obj_to_cstr:")
        return c_str1
    else:
        c_str1 = PyUnicode_AsUTF8AndSize(o1, length)
        if c_str1 == NULL:
            raise OSError("obj_to_cstr:")
    return c_str1
# end IF


cdef inline char* copy_string(  char*   in_str,
                                int     length):
    cdef int i
    cdef char * out_str = <char *> PyMem_Malloc((length+1)*sizeof(char))
    if out_str == NULL:
        raise OSError("Could not allocate memory for sequence.")
    memcpy(out_str, in_str, length+1)
    return out_str

cdef inline void copy_string_buffer(char*   in_str,
                                    char*   out_str,
                                    int     length):
    memcpy(out_str, in_str, length + 1)

