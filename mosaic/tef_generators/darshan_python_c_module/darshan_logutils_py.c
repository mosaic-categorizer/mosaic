#include <Python.h>
#include "./darshan-logutils.h"

void free_mounts(char** mounts, const int mount_count) {
    for(int i = 0; i < mount_count; i++) {
        free(mounts[i]);
    }
    free(mounts);
}

char* get_mount_name(const int64_t dev_id, int64_t* mount_ids, char** mount_names, int count) {
    for(int i = 0; i < count; i++) {
        if(mount_ids[i] == (int64_t)dev_id) {
            return mount_names[i];
        }
    }
    return NULL;
}

PyObject *py_process_log(PyObject *self, PyObject *args)
{
    char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        PyErr_SetString(PyExc_RuntimeError, "could not parse arguments");
        return NULL;
    }

    darshan_fd fd = darshan_log_open(path, "r");

    if (!fd) {
        PyErr_SetString(PyExc_RuntimeError, "could not open file");
        return NULL;
    }

    struct darshan_job job;
    darshan_log_getjob(fd, &job);

    char *job_exe = malloc(CP_EXE_LEN + 1);
    darshan_log_getexe(fd, job_exe);

    int64_t* devs;
    char** mnt_pts;
    char** fs_types;
    int count;
    darshan_log_getmounts(fd, &devs, &mnt_pts, &fs_types, &count);
    char **mounts = malloc(count * sizeof(char*));
    int64_t *mount_ids = malloc(count * sizeof(int64_t));
    for(int i = 0; i < count; i++){
        mount_ids[i] = devs[i];
        mounts[i] = strdup(mnt_pts[i]);
    }
    Py_DECREF(devs);
    Py_DECREF(mnt_pts);
    Py_DECREF(fs_types);

    struct darshan_file file;
    PyObject *operations = PyList_New(0);
    while(1) {
        int read = darshan_log_getfile(fd, &job, &file);
        if(read == 0) {
            break;
        }
        const int64_t rank = file.rank;
        char *mount_res = get_mount_name(file.counters[CP_DEVICE], mount_ids, mounts, count);
        if (mount_res == NULL) {
            free_mounts(mounts, count);
            Py_DECREF(operations);
            PyErr_SetString(PyExc_RuntimeError, "failed to find mount name");
            darshan_log_close(fd);
            return NULL;
        }
        char* mount = strdup(mount_res);
        const double start_ts = file.fcounters[CP_F_OPEN_TIMESTAMP];
        const double end_ts = file.fcounters[CP_F_CLOSE_TIMESTAMP];
        const int64_t opens = file.counters[CP_POSIX_OPENS];
        const int64_t bytes_read = file.counters[CP_BYTES_READ];
        const int64_t bytes_write = file.counters[CP_BYTES_WRITTEN];
        const int64_t seeks = file.counters[CP_POSIX_SEEKS];
        PyObject *py_item = PyDict_New();
        if (!py_item) {
            free(mount);
            free_mounts(mounts, count);
            Py_DECREF(operations);
            PyErr_SetString(PyExc_RuntimeError, "failed to create Python dictionary to store operation");
            darshan_log_close(fd);
            return NULL;
        }

        PyObject *py_rank = PyLong_FromLongLong(rank);
        PyObject *py_mount = PyUnicode_FromString(mount);
        PyObject *py_start = PyFloat_FromDouble(start_ts);
        PyObject *py_end = PyFloat_FromDouble(end_ts);
        PyObject *py_opens = PyLong_FromLongLong(opens);
        PyObject *py_br = PyLong_FromLongLong(bytes_read);
        PyObject *py_bw = PyLong_FromLongLong(bytes_write);
        PyObject *py_seeks = PyLong_FromLongLong(seeks);

        free(mount);

        if (!py_rank || !py_mount || !py_start || !py_end ||
            !py_opens || !py_br || !py_bw || !py_seeks) {
            Py_XDECREF(py_rank); Py_XDECREF(py_mount); Py_XDECREF(py_start);
            Py_XDECREF(py_end); Py_XDECREF(py_opens); Py_XDECREF(py_br);
            Py_XDECREF(py_bw); Py_XDECREF(py_seeks);
            Py_DECREF(py_item);
            Py_DECREF(operations);
            free_mounts(mounts, count);
            PyErr_SetString(PyExc_RuntimeError, "failed to convert C variables to Python objects");
            darshan_log_close(fd);
            return NULL;
        }

        PyDict_SetItemString(py_item, "rank", py_rank);
        PyDict_SetItemString(py_item, "mount", py_mount);
        PyDict_SetItemString(py_item, "start_ts", py_start);
        PyDict_SetItemString(py_item, "end_ts", py_end);
        PyDict_SetItemString(py_item, "opens", py_opens);
        PyDict_SetItemString(py_item, "bytes_read", py_br);
        PyDict_SetItemString(py_item, "bytes_write", py_bw);
        PyDict_SetItemString(py_item, "seeks", py_seeks);

        Py_DECREF(py_rank); Py_DECREF(py_mount); Py_DECREF(py_start);
        Py_DECREF(py_end); Py_DECREF(py_opens); Py_DECREF(py_br);
        Py_DECREF(py_bw); Py_DECREF(py_seeks);

        if (PyList_Append(operations, py_item) < 0) {
            Py_DECREF(py_item);
            Py_DECREF(operations);
            free_mounts(mounts, count);
            PyErr_SetString(PyExc_RuntimeError, "failed to append operation to list");
            darshan_log_close(fd);
            return NULL;
        }
        Py_DECREF(py_item);
    }

    darshan_log_close(fd);

    PyObject *result = PyDict_New();
    if (!result) {
        Py_DECREF(operations);
        free_mounts(mounts, count);
        PyErr_SetString(PyExc_RuntimeError, "failed to create Python dictionary to store trace content");
        return NULL;
    }

    PyDict_SetItemString(result, "uid", PyLong_FromLongLong(job.uid));
    PyDict_SetItemString(result, "pid", PyLong_FromLongLong(job.jobid));
    PyDict_SetItemString(result, "nprocs", PyLong_FromLongLong(job.nprocs));
    PyDict_SetItemString(result, "exe", PyUnicode_FromString(job_exe));
    PyDict_SetItemString(result, "start_ts", PyLong_FromLongLong(job.start_time));
    PyDict_SetItemString(result, "end_ts", PyLong_FromLongLong(job.end_time));
    PyDict_SetItemString(result, "operations", operations);
    Py_DECREF(operations);

    free_mounts(mounts, count);
    return result;
}

static PyMethodDef DarshanMethods[] = {
    {"process_log", (PyCFunction)py_process_log, METH_VARARGS, "Process a darshan log"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef darshan_module = {
    PyModuleDef_HEAD_INIT,
    "darshan_logutils",
    "Darshan v2 log utilities",
    -1,
    DarshanMethods
};

PyMODINIT_FUNC PyInit_darshanv2logutils(void)
{
    return PyModule_Create(&darshan_module);
}