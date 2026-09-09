// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted flag protocol from FreeToken's ple_store_ext.cpp at
// af71ba43206e124f5ff6419b47ee36c6e9981078 (Apache-2.0).
// This implementation uses the CUDA driver directly and translates pinned
// host addresses rather than assuming identical host/device virtual addresses.
#include <Python.h>
#include <cuda.h>
#include <cstdint>

static CUresult device_flag(unsigned long long host, CUdeviceptr* device) {
  return cuMemHostGetDevicePointer(device, reinterpret_cast<void*>(host), 0);
}

static PyObject* signal_flag(PyObject*, PyObject* args) {
  unsigned long long flag;
  if (!PyArg_ParseTuple(args, "K", &flag)) return nullptr;
  __atomic_store_n(reinterpret_cast<uint64_t*>(flag), uint64_t{1},
                   __ATOMIC_RELEASE);
  Py_RETURN_NONE;
}

static PyObject* memop_write(PyObject*, PyObject* args) {
  unsigned long long stream, host, value;
  if (!PyArg_ParseTuple(args, "KKK", &stream, &host, &value)) return nullptr;
  CUdeviceptr flag;
  CUresult status = device_flag(host, &flag);
  if (status == CUDA_SUCCESS)
    status = cuStreamWriteValue64(reinterpret_cast<CUstream>(stream), flag,
                                  value, CU_STREAM_WRITE_VALUE_DEFAULT);
  return PyLong_FromLong(status);
}

static PyObject* memop_wait_geq(PyObject*, PyObject* args) {
  unsigned long long stream, host, value;
  if (!PyArg_ParseTuple(args, "KKK", &stream, &host, &value)) return nullptr;
  CUdeviceptr flag;
  CUresult status = device_flag(host, &flag);
  if (status == CUDA_SUCCESS)
    status = cuStreamWaitValue64(reinterpret_cast<CUstream>(stream), flag,
                                 value, CU_STREAM_WAIT_VALUE_GEQ);
  return PyLong_FromLong(status);
}

static PyObject* memop_wait_reset(PyObject*, PyObject* args) {
  unsigned long long stream, host;
  if (!PyArg_ParseTuple(args, "KK", &stream, &host)) return nullptr;
  CUdeviceptr flag;
  CUresult status = device_flag(host, &flag);
  auto s = reinterpret_cast<CUstream>(stream);
  if (status == CUDA_SUCCESS)
    status = cuStreamWaitValue64(s, flag, 1, CU_STREAM_WAIT_VALUE_GEQ);
  if (status == CUDA_SUCCESS)
    status = cuStreamWriteValue64(s, flag, 0, CU_STREAM_WRITE_VALUE_DEFAULT);
  return PyLong_FromLong(status);
}

static PyMethodDef methods[] = {
    {"signal_flag", signal_flag, METH_VARARGS, "Release-store the host flag."},
    {"memop_write", memop_write, METH_VARARGS, "Queue a 64-bit flag write."},
    {"memop_wait_geq", memop_wait_geq, METH_VARARGS, "Queue a flag wait."},
    {"memop_wait_reset", memop_wait_reset, METH_VARARGS,
     "Queue wait and reset."},
    {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_ple_memops", nullptr, -1,
                             methods};
PyMODINIT_FUNC PyInit__ple_memops() { return PyModule_Create(&module); }
