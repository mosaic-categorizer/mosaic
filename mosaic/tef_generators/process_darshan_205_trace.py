import os
from ctypes import *
from enum import Enum
import pathlib


class DarshanIndices(Enum):
    CP_INDEP_OPENS = 0
    CP_COLL_OPENS = 1
    CP_INDEP_READS = 2
    CP_INDEP_WRITES = 3
    CP_COLL_READS = 4
    CP_COLL_WRITES = 5
    CP_SPLIT_READS = 6
    CP_SPLIT_WRITES = 7
    CP_NB_READS = 8
    CP_NB_WRITES = 9
    CP_SYNCS = 10
    CP_POSIX_READS = 11
    CP_POSIX_WRITES = 12
    CP_POSIX_OPENS = 13
    CP_POSIX_SEEKS = 14
    CP_POSIX_STATS = 15
    CP_POSIX_MMAPS = 16
    CP_POSIX_FREADS = 17
    CP_POSIX_FWRITES = 18
    CP_POSIX_FOPENS = 19
    CP_POSIX_FSEEKS = 20
    CP_POSIX_FSYNCS = 21
    CP_POSIX_FDSYNCS = 22
    CP_INDEP_NC_OPENS = 23
    CP_COLL_NC_OPENS = 24
    CP_HDF5_OPENS = 25
    CP_COMBINER_NAMED = 26
    CP_COMBINER_DUP = 27
    CP_COMBINER_CONTIGUOUS = 28
    CP_COMBINER_VECTOR = 29
    CP_COMBINER_HVECTOR_INTEGER = 30
    CP_COMBINER_HVECTOR = 31
    CP_COMBINER_INDEXED = 32
    CP_COMBINER_HINDEXED_INTEGER = 33
    CP_COMBINER_HINDEXED = 34
    CP_COMBINER_INDEXED_BLOCK = 35
    CP_COMBINER_STRUCT_INTEGER = 36
    CP_COMBINER_STRUCT = 37
    CP_COMBINER_SUBARRAY = 38
    CP_COMBINER_DARRAY = 39
    CP_COMBINER_F90_REAL = 40
    CP_COMBINER_F90_COMPLEX = 41
    CP_COMBINER_F90_INTEGER = 42
    CP_COMBINER_RESIZED = 43
    CP_HINTS = 44
    CP_VIEWS = 45
    CP_MODE = 46
    CP_BYTES_READ = 47
    CP_BYTES_WRITTEN = 48
    CP_MAX_BYTE_READ = 49
    CP_MAX_BYTE_WRITTEN = 50
    CP_CONSEC_READS = 51
    CP_CONSEC_WRITES = 52
    CP_SEQ_READS = 53
    CP_SEQ_WRITES = 54
    CP_RW_SWITCHES = 55
    CP_MEM_NOT_ALIGNED = 56
    CP_MEM_ALIGNMENT = 57
    CP_FILE_NOT_ALIGNED = 58
    CP_FILE_ALIGNMENT = 59
    CP_MAX_READ_TIME_SIZE = 60
    CP_MAX_WRITE_TIME_SIZE = 61
    CP_SIZE_READ_0_100 = 62
    CP_SIZE_READ_100_1K = 63
    CP_SIZE_READ_1K_10K = 64
    CP_SIZE_READ_10K_100K = 65
    CP_SIZE_READ_100K_1M = 66
    CP_SIZE_READ_1M_4M = 67
    CP_SIZE_READ_4M_10M = 68
    CP_SIZE_READ_10M_100M = 69
    CP_SIZE_READ_100M_1G = 70
    CP_SIZE_READ_1G_PLUS = 71
    CP_SIZE_WRITE_0_100 = 72
    CP_SIZE_WRITE_100_1K = 73
    CP_SIZE_WRITE_1K_10K = 74
    CP_SIZE_WRITE_10K_100K = 75
    CP_SIZE_WRITE_100K_1M = 76
    CP_SIZE_WRITE_1M_4M = 77
    CP_SIZE_WRITE_4M_10M = 78
    CP_SIZE_WRITE_10M_100M = 79
    CP_SIZE_WRITE_100M_1G = 80
    CP_SIZE_WRITE_1G_PLUS = 81
    CP_SIZE_READ_AGG_0_100 = 82
    CP_SIZE_READ_AGG_100_1K = 83
    CP_SIZE_READ_AGG_1K_10K = 84
    CP_SIZE_READ_AGG_10K_100K = 85
    CP_SIZE_READ_AGG_100K_1M = 86
    CP_SIZE_READ_AGG_1M_4M = 87
    CP_SIZE_READ_AGG_4M_10M = 88
    CP_SIZE_READ_AGG_10M_100M = 89
    CP_SIZE_READ_AGG_100M_1G = 90
    CP_SIZE_READ_AGG_1G_PLUS = 91
    CP_SIZE_WRITE_AGG_0_100 = 92
    CP_SIZE_WRITE_AGG_100_1K = 93
    CP_SIZE_WRITE_AGG_1K_10K = 94
    CP_SIZE_WRITE_AGG_10K_100K = 95
    CP_SIZE_WRITE_AGG_100K_1M = 96
    CP_SIZE_WRITE_AGG_1M_4M = 97
    CP_SIZE_WRITE_AGG_4M_10M = 98
    CP_SIZE_WRITE_AGG_10M_100M = 99
    CP_SIZE_WRITE_AGG_100M_1G = 100
    CP_SIZE_WRITE_AGG_1G_PLUS = 101
    CP_EXTENT_READ_0_100 = 102
    CP_EXTENT_READ_100_1K = 103
    CP_EXTENT_READ_1K_10K = 104
    CP_EXTENT_READ_10K_100K = 105
    CP_EXTENT_READ_100K_1M = 106
    CP_EXTENT_READ_1M_4M = 107
    CP_EXTENT_READ_4M_10M = 108
    CP_EXTENT_READ_10M_100M = 109
    CP_EXTENT_READ_100M_1G = 110
    CP_EXTENT_READ_1G_PLUS = 111
    CP_EXTENT_WRITE_0_100 = 112
    CP_EXTENT_WRITE_100_1K = 113
    CP_EXTENT_WRITE_1K_10K = 114
    CP_EXTENT_WRITE_10K_100K = 115
    CP_EXTENT_WRITE_100K_1M = 116
    CP_EXTENT_WRITE_1M_4M = 117
    CP_EXTENT_WRITE_4M_10M = 118
    CP_EXTENT_WRITE_10M_100M = 119
    CP_EXTENT_WRITE_100M_1G = 120
    CP_EXTENT_WRITE_1G_PLUS = 121
    CP_STRIDE1_STRIDE = 122
    CP_STRIDE2_STRIDE = 123
    CP_STRIDE3_STRIDE = 124
    CP_STRIDE4_STRIDE = 125
    CP_STRIDE1_COUNT = 126
    CP_STRIDE2_COUNT = 127
    CP_STRIDE3_COUNT = 128
    CP_STRIDE4_COUNT = 129
    CP_ACCESS1_ACCESS = 130
    CP_ACCESS2_ACCESS = 131
    CP_ACCESS3_ACCESS = 132
    CP_ACCESS4_ACCESS = 133
    CP_ACCESS1_COUNT = 134
    CP_ACCESS2_COUNT = 135
    CP_ACCESS3_COUNT = 136
    CP_ACCESS4_COUNT = 137
    CP_DEVICE = 138
    CP_SIZE_AT_OPEN = 139
    CP_FASTEST_RANK = 140
    CP_FASTEST_RANK_BYTES = 141
    CP_SLOWEST_RANK = 142
    CP_SLOWEST_RANK_BYTES = 143


class FDarshanIndices(Enum):
    CP_F_OPEN_TIMESTAMP = 0
    CP_F_READ_START_TIMESTAMP = 1
    CP_F_WRITE_START_TIMESTAMP = 2
    CP_F_CLOSE_TIMESTAMP = 3
    CP_F_READ_END_TIMESTAMP = 4
    CP_F_WRITE_END_TIMESTAMP = 5
    CP_F_POSIX_READ_TIME = 6
    CP_F_POSIX_WRITE_TIME = 7
    CP_F_POSIX_META_TIME = 8
    CP_F_MPI_META_TIME = 9
    CP_F_MPI_READ_TIME = 10
    CP_F_MPI_WRITE_TIME = 11
    CP_F_MAX_READ_TIME = 12
    CP_F_MAX_WRITE_TIME = 13
    CP_F_FASTEST_RANK_TIME = 14
    CP_F_SLOWEST_RANK_TIME = 15
    CP_F_VARIANCE_RANK_TIME = 16
    CP_F_VARIANCE_RANK_BYTES = 17


class DarshanJob(Structure):
    _fields_ = [
        ("version_string", c_char * 8),
        ("magic_nr", c_int64),
        ("uid", c_int64),
        ("start_time", c_int64),
        ("end_time", c_int64),
        ("nprocs", c_int64),
        ("jobid", c_int64),
        ("metadata", c_char * 1024),
    ]


class DarshanFDS(Structure):
    _fields_ = [
        ("gzf", c_void_p),
        ("pos", c_int64),
        ("mode", c_char * 2),
        ("swap_flag", c_int),
        ("version", c_char * 10),
        ("job_struct_size", c_int),
        ("name", POINTER(c_char)),
        ("COMPAT_CP_EXE_LEN", c_int),
    ]


class DarshanFile(Structure):
    _fields_ = [
        ("hash", c_uint64),
        ("rank", c_int64),
        ("name_suffix", c_char * 16),
        ("counters", c_int64 * 144),
        ("fcounters", c_double * 18),
    ]


def read_darshan_205(trace: str, mount: str = "/") -> dict:
    shared_lib_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "utils/darshan-logutils.so"
    )

    if not trace.endswith(".darshan"):
        raise Exception("Not a valid darshan trace")

    if not os.path.isfile(trace):
        raise Exception("Trace not found")

    with open(trace, "rb") as f:
        version = os.pread(f.fileno(), 4, 0).decode("UTF-8")
        if version != "2.05":
            raise Exception(f"Unsupported version: {version}")

    darshan_df = POINTER(DarshanFDS)

    logutils = CDLL(shared_lib_path)
    logutils.darshan_log_open.restype = darshan_df

    fd = logutils.darshan_log_open(trace.encode("UTF-8"), "r")
    job = DarshanJob()
    logutils.darshan_log_getjob(fd, byref(job))

    log_exe = (c_char * 4096)()
    logutils.darshan_log_getexe(fd, byref(log_exe))

    devs = POINTER(c_int64)()
    mnt_pts = POINTER(POINTER(c_char))()
    fs_types = POINTER(POINTER(c_char))()
    mount_count = c_int()
    logutils.darshan_log_getmounts(
        fd, byref(devs), byref(mnt_pts), byref(fs_types), byref(mount_count)
    )
    mount_dict = {}
    for i in range(mount_count.value):
        mount_dict[int(devs[i])] = string_at(mnt_pts[i]).decode("UTF-8"), string_at(
            fs_types[i]
        ).decode("UTF-8")

    go_next = 1
    darshan_file = DarshanFile()
    operations = []
    while go_next:
        go_next = logutils.darshan_log_getfile(fd, byref(job), byref(darshan_file))
        if go_next:
            opens = darshan_file.counters[DarshanIndices.CP_POSIX_OPENS.value]
            read = darshan_file.counters[DarshanIndices.CP_BYTES_READ.value]
            write = darshan_file.counters[DarshanIndices.CP_BYTES_WRITTEN.value]
            seeks = darshan_file.counters[DarshanIndices.CP_POSIX_SEEKS.value]
            if opens == read == write == seeks == 0 or not mount_dict[
                darshan_file.counters[DarshanIndices.CP_DEVICE.value]
            ][0].startswith(mount):
                continue
            operations.append(
                {
                    "rank": darshan_file.rank,
                    "mount": mount_dict[
                        darshan_file.counters[DarshanIndices.CP_DEVICE.value]
                    ][0],
                    "start_ts": darshan_file.fcounters[
                        FDarshanIndices.CP_F_OPEN_TIMESTAMP.value
                    ],
                    "end_ts": darshan_file.fcounters[
                        FDarshanIndices.CP_F_CLOSE_TIMESTAMP.value
                    ],
                    "opens": opens,
                    "bytes_read": read,
                    "bytes_write": write,
                    "seeks": seeks,
                }
            )
    logutils.darshan_log_close(fd)

    return {
        "uid": job.uid,
        "pid": job.jobid,
        "nprocs": job.nprocs,
        "exe": string_at(log_exe.value).decode("UTF-8"),
        "start_ts": job.start_time,
        "end_ts": job.end_time,
        "operations": operations,
    }
