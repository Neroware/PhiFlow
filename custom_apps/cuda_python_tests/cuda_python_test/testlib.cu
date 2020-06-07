#include "cuda.h"
#include "cuda_runtime.h"
#include <Python.h>


/*int Cfib(int n){
        if(n < 2)
                return n;
        else
                return Cfib(n - 1) + Cfib(n - 2);
}*/


/*static PyObject* fib(PyObject* self, PyObject* args){
        int n;

        if(!PyArg_ParseTuple(args, "i", &n))
                return NULL;

        return Py_BuildValue("i", Cfib(n));
}*/


static PyObject* version(PyObject* self){
        return Py_BuildValue("s", "Version 1.0");
}


static PyMethodDef methods[] = {
        //{"fib", fib, METH_VARARGS, "Calculates Fibonacci Numbers."},
        {"version", (PyCFunction) version, METH_NOARGS, "Version Number."},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef testlib = {
        PyModuleDef_HEAD_INIT,
        "testlib",
        "Test-Lib containing Fibonacci-Function",
        -1,
        methods
};


PyMODINIT_FUNC PyInit_testlib(void)
{
        return PyModule_Create(&testlib);
}
