#include <Python.h>

#include "sharedlock.h"

PyObject* size(PyObject* self) {
    long size = sharedlock_size();
    return PyLong_FromUnsignedLong(size);
}

PyObject* create(PyObject* self, PyObject* args) {
    Py_buffer arg;
    if (!PyArg_ParseTuple(args, "y*", &arg)) {
        return NULL;
    }
    sharedlock_create((char*)arg.buf);
    return Py_None;
}

PyObject* acquire(PyObject* self, PyObject* args) {
    Py_buffer arg;
    if (!PyArg_ParseTuple(args, "y*", &arg)) {
        return NULL;
    }
    sharedlock_acquire((char*)arg.buf);
    return Py_None;
}

PyObject* release(PyObject* self, PyObject* args) {
    Py_buffer arg;
    if (!PyArg_ParseTuple(args, "y*", &arg)) {
        return NULL;
    }
    sharedlock_release((char*)arg.buf);
    return Py_None;
}

PyObject* destroy(PyObject* self, PyObject* args) {
    Py_buffer arg;
    if (!PyArg_ParseTuple(args, "y*", &arg)) {
        return NULL;
    }
    sharedlock_destroy((char*)arg.buf);
    return Py_None;
}

PyMethodDef locking_funcs[] = {
    {
        "size",
        (PyCFunction)size,
        METH_NOARGS,
        "DOCS",
    },
    {
        "create",
        (PyCFunction)create,
        METH_VARARGS,
        "DOCS",
    },
    {
        "acquire",
        (PyCFunction)acquire,
        METH_VARARGS,
        "DOCS",
    },
    {
        "release",
        (PyCFunction)release,
        METH_VARARGS,
        "DOCS",
    },
    {
        "destroy",
        (PyCFunction)destroy,
        METH_VARARGS,
        "DOCS",
    },
    {
        NULL
    },
};

char locking_docs[] = "These are the locking docs.";

PyModuleDef locking_mod = {
    PyModuleDef_HEAD_INIT,
    "locking",
    locking_docs,
    -1,
    locking_funcs,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_locking(void) {
    return PyModule_Create(&locking_mod);
}
