import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from darshan.report import DarshanReport
from tqdm import tqdm

from mosaic.process_pool import ProcessPool
from mosaic import darshanv2logutils


def export_metadata(report: DarshanReport, name: str) -> dict:
    return {
        "file": name,
        "uid": report.metadata["job"]["uid"],
        "job_id": report.metadata["job"]["jobid"],
        "start_ts": datetime.fromtimestamp(
            report.metadata["job"]["start_time_sec"]
            + report.metadata["job"]["start_time_nsec"] / 1e9
        ).timestamp(),
        "end_ts": datetime.fromtimestamp(
            report.metadata["job"]["end_time_sec"]
            + report.metadata["job"]["end_time_nsec"] / 1e9
        ).timestamp(),
        "run_time": report.metadata["job"]["run_time"],
        "exe": report.metadata["exe"],
        "n_nodes": report.metadata["job"]["nprocs"],
    }


def export_operations(
        report: DarshanReport, pid: int, node_count: int, mount: str
) -> list:
    res = []
    start_ts = datetime.fromtimestamp(
        report.metadata["job"]["start_time_sec"]
        + report.metadata["job"]["start_time_nsec"] / 1e9
    )
    dxt_modules = set(filter(lambda m: m.startswith("DXT_"), report.records.keys()))
    standard_modules = set(
        filter(lambda m: m in ["POSIX", "STDIO"], report.records.keys())
    )
    for module in dxt_modules:
        res.extend(
            export_ops_of_dxt_module(report, module, pid, node_count, start_ts, mount)
        )
        standard_modules = set(
            filter(lambda m: m != module.replace("DXT_", ""), standard_modules)
        )
    for module in standard_modules:
        res.extend(
            export_ops_of_standard_module(report, module, pid, node_count, mount)
        )
    return res


def export_legacy_operations(
        t_content: dict, pid: int, node_count: int, mount: str
) -> list:
    res = []
    start_ts = datetime.fromtimestamp(t_content["start_ts"])
    for op in t_content["operations"]:
        if not op["mount"].startswith(mount):
            continue
        rank = op["rank"]
        op_start = (start_ts + timedelta(seconds=op["start_ts"])).timestamp()
        op_duration = op["end_ts"] - op["start_ts"]
        opens = int(
            op["opens"]
            / (
                (node_count if rank == -1 else 1) * 2
                if op["bytes_read"] and op["bytes_write"]
                else 1
            )
        )
        seeks = int(
            op["seeks"]
            / (
                (node_count if rank == -1 else 1) * 2
                if op["bytes_read"] and op["bytes_write"]
                else 1
            )
        )
        if op["bytes_read"]:
            amount = int(op["bytes_read"] / (node_count if rank == -1 else 1))
            for i in ([rank] if rank != -1 else range(1, node_count)):
                res.append(
                    generate_op_struc(
                        "read",
                        "POSIX",
                        pid,
                        i,
                        op_start,
                        op_duration,
                        amount,
                        op["mount"],
                        opens=opens,
                        seeks=seeks,
                    )
                )
        if op["bytes_write"]:
            amount = int(op["bytes_write"] / (node_count if rank == -1 else 1))
            for i in ([rank] if rank != -1 else range(1, node_count)):
                res.append(
                    generate_op_struc(
                        "write",
                        "POSIX",
                        pid,
                        i,
                        op_start,
                        op_duration,
                        amount,
                        op["mount"],
                        opens=opens,
                        seeks=seeks,
                    )
                )
    return res


def export_ops_of_standard_module(
        report: DarshanReport, module: str, pid: int, node_count: int, mount: str
) -> list:
    res = []
    module_df = pd.merge(
        report.records[module].to_df()["counters"],
        report.records[module].to_df()["fcounters"],
        left_on=["id", "rank"],
        right_on=["id", "rank"],
        how="inner",
        validate="many_to_many",
    )
    module_df["filename"] = module_df["id"].apply(lambda i: report.name_records[i])
    start_ts = datetime.fromtimestamp(
        report.metadata["job"]["start_time_sec"]
        + report.metadata["job"]["start_time_nsec"] / 1e9
    )
    for _, op in module_df.iterrows():
        if not op["filename"].startswith(mount):
            continue
        if op[f"{module}_BYTES_READ"]:
            res.extend(parse_standard_op(op, module, "read", node_count, start_ts, pid))
        if op[f"{module}_BYTES_WRITTEN"]:
            res.extend(
                parse_standard_op(op, module, "write", node_count, start_ts, pid)
            )
    return res


def export_ops_of_dxt_module(
        report: DarshanReport,
        module: str,
        pid: int,
        node_count: int,
        start_ts: datetime,
        mount: str,
) -> list:
    res = []
    for f in report.records[module].to_df():
        filename = report.name_records[f["id"]]
        if not filename.startswith(mount):
            continue
        rank = f["rank"]
        if not f["write_segments"].empty:
            write_segments = f["write_segments"].query("length > 0")
            for _, op in write_segments.iterrows():
                res.extend(
                    parse_dxt_op(
                        op, module, "write", filename, rank, node_count, start_ts, pid
                    )
                )
        if not f["read_segments"].empty:
            read_segments = f["read_segments"].query("length > 0")
            for _, op in read_segments.iterrows():
                res.extend(
                    parse_dxt_op(
                        op, module, "read", filename, rank, node_count, start_ts, pid
                    )
                )
    return res


def parse_standard_op(
        op, module: str, mode: str, node_count: int, start_ts: datetime, pid: int
) -> list:
    mode_up = mode.upper()
    mode_passive = "WRITTEN" if mode == "write" else mode_up
    rank = op["rank"]
    op_duration = float(
        op[f"{module}_F_{mode_up}_END_TIMESTAMP"]
        - op[f"{module}_F_{mode_up}_START_TIMESTAMP"]
    )
    amount = int(
        op[f"{module}_BYTES_{mode_passive}"] / (node_count if rank == -1 else 1)
    )
    op_start = (
            start_ts + timedelta(seconds=op[f"{module}_F_{mode_up}_START_TIMESTAMP"])
    ).timestamp()
    opens = int(
        op[f"{module}_OPENS"]
        / (
            (node_count if rank == -1 else 1) * 2
            if op[f"{module}_BYTES_READ"] and op[f"{module}_BYTES_WRITTEN"]
            else 1
        )
    )
    seeks = int(
        op[f"{module}_SEEKS"]
        / (
            (node_count if rank == -1 else 1) * 2
            if op[f"{module}_BYTES_READ"] and op[f"{module}_BYTES_WRITTEN"]
            else 1
        )
    )
    operations = []
    for i in ([rank] if rank != -1 else range(1, node_count)):
        operations.append(
            generate_op_struc(
                mode,
                module,
                pid,
                i,
                op_start,
                op_duration,
                amount,
                op["filename"],
                opens=opens,
                seeks=seeks,
            )
        )
    return operations


def parse_dxt_op(
        op,
        module: str,
        mode: str,
        filename: str,
        rank: int,
        node_count: int,
        start_ts: datetime,
        pid: int,
) -> list:
    op_duration = float(op["end_time"] - op["start_time"])
    amount = int(op["length"] / (node_count if rank == -1 else 1))
    op_start = (start_ts + timedelta(seconds=op["start_time"])).timestamp()
    offset = op["offset"]
    operations = []
    for i in ([rank] if rank != -1 else range(1, node_count)):
        operations.append(
            generate_op_struc(
                mode,
                module,
                pid,
                i,
                op_start,
                op_duration,
                amount,
                filename,
                offset=offset,
            )
        )
    return operations


def generate_op_struc(
        mode: str,
        cat: str,
        pid: int,
        tid: int,
        op_start: float,
        op_dur: float,
        amount: int,
        filename: str,
        opens: int = None,
        seeks: int = None,
        offset: int = None,
) -> dict:
    if op_dur == 0:
        raise RuntimeError("null operation duration")
    return {
        "name": mode,
        "cat": cat,
        "pid": pid,
        "tid": tid,
        "ts": op_start * 1e6,
        "dur": op_dur * 1e6,
        "ph": "X",
        "args": {
            "count": amount,
            "file": filename,
            "speed": amount / op_dur,
            "opens": opens,
            "seeks": seeks,
            "offset": offset,
        },
    }


def generate_op_metadata(pid: int, node_count: int) -> list:
    res = [{"name": "process_name", "ph": "M", "pid": pid, "args": {"name": "Job"}}]
    for i in range(node_count):
        res.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": i,
                "args": {"name": "Rank"},
            }
        )
    return res


def generate_trace_event_json(trace: str, output_directory: str, mount: str = "/"):
    if os.path.isfile(
            os.path.join(output_directory, trace.split("/")[-1] + ".tef.json.gz")
    ):
        return ""
    trace_name = trace.split("/")[-1]
    try:
        version = ""
        with open(trace, "rb") as f:
            version = os.pread(f.fileno(), 4, 0).decode("UTF-8")
        if version == "2.05":
            return generate_legacy_trace_event_json(trace, output_directory, mount)
        else:
            report = DarshanReport(trace, read_all=True)
        metadata = export_metadata(report, trace_name)
        pid = metadata["job_id"]
        operations = export_operations(report, pid, metadata["n_nodes"], mount)
        operations.extend(generate_op_metadata(pid, metadata["n_nodes"]))
        operations.append(
            {
                "name": "Darshan_start",
                "ph": "i",
                "ts": metadata["start_ts"] * 1e6,
                "pid": 0,
                "tid": 0,
                "s": "g",
            }
        )
        operations.append(
            {
                "name": "Darshan_end",
                "ph": "i",
                "ts": metadata["end_ts"] * 1e6,
                "pid": 0,
                "tid": 0,
                "s": "g",
            }
        )
        perfetto_trace = {
            "traceEvents": operations,
            "metadata": metadata,
        }
    except Exception as e:
        print(f"error {trace}: {e}", file=sys.stderr)
        return f"error {trace}: {e}"
    with gzip.open(
            os.path.join(output_directory, trace.split("/")[-1] + ".tef.json.gz"), "wt"
    ) as f:
        json.dump(perfetto_trace, f, indent=2)
    return ""


def generate_legacy_trace_event_json(
        trace: str, output_directory: str, mount: str = "/"
):
    trace_name = trace.split("/")[-1]
    trace_content = darshanv2logutils.process_log(trace)
    trace_content["operations"] = [
        op for op in trace_content["operations"] if op["mount"].startswith(mount)
    ]
    operations = export_legacy_operations(
        trace_content, trace_content["pid"], trace_content["nprocs"], mount
    )
    operations.extend(
        generate_op_metadata(trace_content["pid"], trace_content["nprocs"])
    )
    operations.append(
        {
            "name": "Darshan_start",
            "ph": "i",
            "ts": trace_content["start_ts"] * 1e6,
            "pid": 0,
            "tid": 0,
            "s": "g",
        }
    )
    operations.append(
        {
            "name": "Darshan_end",
            "ph": "i",
            "ts": trace_content["start_ts"] * 1e6,
            "pid": 0,
            "tid": 0,
            "s": "g",
        }
    )
    perfetto_trace = {
        "traceEvents": operations,
        "metadata": {
            "file": trace_name,
            "uid": trace_content["uid"],
            "job_id": trace_content["pid"],
            "start_ts": trace_content["start_ts"],
            "end_ts": trace_content["end_ts"],
            "run_time": trace_content["end_ts"] - trace_content["start_ts"],
            "exe": trace_content["exe"],
            "n_nodes": trace_content["nprocs"],
        },
    }
    with gzip.open(
            os.path.join(output_directory, trace.split("/")[-1] + ".tef.json.gz"), "wt"
    ) as f:
        json.dump(perfetto_trace, f, indent=2)
    return ""


def generate_traces_from_directory(
        darshan_directory: str, output_directory: str, mount: str = "/", cpu_count: int = -1
):
    traces_to_convert = []
    for file in os.listdir(darshan_directory):
        if file.endswith(".darshan"):
            traces_to_convert.append(os.path.join(darshan_directory, file))
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    process_pool = ProcessPool(cpu_count if cpu_count != -1 else (os.cpu_count() - 1))
    process_pool.batch_submit(
        traces_to_convert, generate_trace_event_json, output_directory, mount
    )
    process_pool.wait_completion()
    for result in process_pool.get_result():
        if result.startswith("error"):
            with open(os.path.join(output_directory, "errors.txt"), "a") as f:
                f.write(result + "\n")


def get_merged_tef_name(tef: dict) -> str:
    user = tef["metadata"]["uid"]
    job_id = tef["metadata"]["job_id"]
    start_ts = tef["metadata"]["start_ts"]
    exe = tef["metadata"]["exe"].split(" ")[0].replace("/", "").replace(".", "")
    return f"{user}_{job_id}_{start_ts}_{exe}.tef.json.gz"


def merge_tef_files_same_job(tef_directory: str, out_directory: str):
    traces_per_job = {}
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)
    for file in os.listdir(tef_directory):
        if file.endswith(".tef.json.gz"):
            with gzip.open(os.path.join(tef_directory, file), "rt") as f:
                job_id = json.load(f)["metadata"]["job_id"]
            if job_id not in traces_per_job:
                traces_per_job[job_id] = []
            traces_per_job[job_id].append(os.path.join(tef_directory, file))
    jobs = traces_per_job.keys()
    for job in tqdm(jobs):
        if len(traces_per_job[job]) < 2:
            with gzip.open(os.path.join(tef_directory, traces_per_job[job][0]), "rt") as f:
                tef = json.load(f)
            shutil.copy(os.path.join(tef_directory, traces_per_job[job][0]),
                        os.path.join(out_directory, get_merged_tef_name(tef)))
            continue
        with gzip.open(os.path.join(tef_directory, traces_per_job[job][0]), "rt") as f:
            trace = json.load(f)
        metadata = {
            "file": trace["metadata"]["file"],
            "uid": trace["metadata"]["uid"],
            "job_id": trace["metadata"]["job_id"],
            "start_ts": 0,
            "end_ts": 0,
            "run_time": 0,
            "exe": trace["metadata"]["exe"],
            "n_nodes": trace["metadata"]["n_nodes"],
        }
        earliest_start = trace["metadata"]["start_ts"]
        latest_end = trace["metadata"]["end_ts"]
        operations = [op for op in trace["traceEvents"] if op["ph"] in ["M", "X"]]
        for trace_name in traces_per_job[job][1:]:
            with gzip.open(os.path.join(tef_directory, trace_name), "rt") as f:
                trace = json.load(f)
            if trace["metadata"]["n_nodes"] != metadata["n_nodes"]:
                print("Node count missmatch")
            earliest_start = min(earliest_start, trace["metadata"]["start_ts"])
            latest_end = max(latest_end, trace["metadata"]["end_ts"])
            operations.extend([op for op in trace["traceEvents"] if op["ph"] in ["X"]])
        operations.sort(key=lambda op: op["ts"] if "ts" in op else 0)
        operations.append(
            {
                "name": "Darshan_start",
                "ph": "i",
                "ts": earliest_start * 1e6,
                "pid": 0,
                "tid": 0,
                "s": "g",
            }
        )
        operations.append(
            {
                "name": "Darshan_end",
                "ph": "i",
                "ts": latest_end * 1e6,
                "pid": 0,
                "tid": 0,
                "s": "g",
            }
        )
        metadata["start_ts"] = earliest_start
        metadata["end_ts"] = latest_end
        metadata["run_time"] = latest_end - earliest_start
        perfetto_trace = {
            "traceEvents": operations,
            "metadata": metadata,
        }
        with gzip.open(os.path.join(out_directory, get_merged_tef_name(perfetto_trace)), "wt") as f:
            json.dump(perfetto_trace, f, indent=2)
