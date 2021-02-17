import cython
from cpython.pycapsule cimport *

cdef extern from "add.h":
    void cadd(void* obj, float value)

def add(obj, float value):
    cadd(PyCapsule_GetPointer(obj, "dltensor"), value)

