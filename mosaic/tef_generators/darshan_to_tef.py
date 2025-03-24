import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from darshan.report import DarshanReport

from mosaic.process_pool import ProcessPool


def export_metadata(report: DarshanReport, name: str) -> dict:
    return {
        'file': name,
        'uid': report.metadata['job']['uid'],
        'job_id': report.metadata['job']['jobid'],
        'start_ts': datetime.fromtimestamp(
            report.metadata['job']['start_time_sec'] + report.metadata['job']['start_time_nsec'] / 1e9).timestamp(),
        'end_ts': datetime.fromtimestamp(
            report.metadata['job']['end_time_sec'] + report.metadata['job']['end_time_nsec'] / 1e9).timestamp(),
        'run_time': report.metadata['job']['run_time'],
        'exe': report.metadata['exe'],
        'n_nodes': report.metadata['job']['nprocs']
    }


def export_operations(report: DarshanReport, pid: int, node_count: int, mount: str) -> list:
    res = []
    start_ts = datetime.fromtimestamp(
        report.metadata['job']['start_time_sec'] + report.metadata['job']['start_time_nsec'] / 1e9)
    dxt_modules = set(filter(lambda m: m.startswith('DXT_'), report.records.keys()))
    standard_modules = set(filter(lambda m: m in ['POSIX', 'STDIO'], report.records.keys()))
    for module in dxt_modules:
        res.extend(export_ops_of_dxt_module(report, module, pid, node_count, start_ts, mount))
        standard_modules = set(filter(lambda m: m != module.replace('DXT_', ''), standard_modules))
    for module in standard_modules:
        res.extend(export_ops_of_standard_module(report, module, pid, node_count, mount))
    return res


def export_ops_of_standard_module(report: DarshanReport, module: str, pid: int, node_count: int, mount: str) -> list:
    res = []
    module_df = pd.merge(report.records[module].to_df()['counters'],
                         report.records[module].to_df()['fcounters'],
                         left_on=['id', 'rank'], right_on=['id', 'rank'], how="inner", validate="many_to_many")
    module_df['filename'] = module_df['id'].apply(lambda i: report.name_records[i])
    start_ts = datetime.fromtimestamp(
        report.metadata['job']['start_time_sec'] + report.metadata['job']['start_time_nsec'] / 1e9)
    for _, op in module_df.iterrows():
        if not op['filename'].startswith(mount):
            continue
        if op[f'{module}_BYTES_READ']:
            res.extend(parse_standard_op(op, module, 'read', node_count, start_ts, pid))
        if op[f'{module}_BYTES_WRITTEN']:
            res.extend(parse_standard_op(op, module, 'write', node_count, start_ts, pid))
    return res


def export_ops_of_dxt_module(report: DarshanReport, module: str, pid: int, node_count: int, start_ts: datetime,
                             mount: str) -> list:
    res = []
    for f in report.records[module].to_df():
        filename = report.name_records[f['id']]
        if not filename.startswith(mount):
            continue
        rank = f['rank']
        if not f['write_segments'].empty:
            write_segments = f['write_segments'].query('length > 0')
            for _, op in write_segments.iterrows():
                res.extend(parse_dxt_op(op, module, 'write', filename, rank, node_count, start_ts, pid))
        if not f['read_segments'].empty:
            read_segments = f['read_segments'].query('length > 0')
            for _, op in read_segments.iterrows():
                res.extend(parse_dxt_op(op, module, 'read', filename, rank, node_count, start_ts, pid))
    return res


def parse_standard_op(op, module: str, mode: str, node_count: int, start_ts: datetime, pid: int) -> list:
    mode_up = mode.upper()
    mode_passive = 'WRITTEN' if mode == 'write' else mode_up
    rank = op['rank']
    op_duration = float(max(op[f'{module}_F_{mode_up}_END_TIMESTAMP'] - op[f'{module}_F_{mode_up}_START_TIMESTAMP'], 1e-6))
    amount = int(op[f'{module}_BYTES_{mode_passive}'] / (node_count if rank == -1 else 1))
    op_start = (start_ts + timedelta(seconds=op[f'{module}_F_{mode_up}_START_TIMESTAMP'])).timestamp()
    opens = op[f'{module}_OPENS']
    seeks = op[f'{module}_SEEKS']
    operations = []
    for i in ([rank] if rank != -1 else range(1, node_count)):
        operations.append(
            generate_op_struc(mode, module, pid, i, op_start, op_duration, amount, op['filename'], opens=opens,
                              seeks=seeks))
    return operations


def parse_dxt_op(op, module: str, mode: str, filename: str, rank: int, node_count: int, start_ts: datetime,
                 pid: int) -> list:
    op_duration = float(op['end_time'] - op['start_time'])
    amount = int(op['length'] / (node_count if rank == -1 else 1))
    op_start = (start_ts + timedelta(seconds=op['start_time'])).timestamp()
    offset = op['offset']
    operations = []
    for i in ([rank] if rank != -1 else range(1, node_count)):
        operations.append(
            generate_op_struc(mode, module, pid, i, op_start, op_duration, amount, filename, offset=offset))
    return operations


def generate_op_struc(mode: str, cat: str, pid: int, tid: int, op_start: float, op_dur: float, amount: int,
                      filename: str, opens: int = None, seeks: int = None, offset: int = None) -> dict:
    return {
        'name': mode,
        'cat': cat,
        'pid': pid,
        'tid': tid,
        'ts': op_start * 1e6,
        'dur': op_dur * 1e6,
        'ph': 'X',
        'args': {
            'count': amount,
            'file': filename,
            'speed': amount / op_dur,
            'opens': opens,
            'seeks': seeks,
            'offset': offset,
        }
    }


def generate_op_metadata(pid: int, node_count: int) -> list:
    res = [{"name": "process_name", "ph": "M", "pid": pid, "args": {"name": 'Job'}}]
    for i in range(node_count):
        res.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": i, "args": {"name": "Rank"}})
    return res


def generate_trace_event_json(trace: str, output_directory: str, mount: str = '/'):
    try:
        report = DarshanReport(trace, read_all=True)
    except Exception as e:
        return f'error {trace}: {e}'
    trace_name = trace.split('/')[-1]
    metadata = export_metadata(report, trace_name)
    pid = metadata['job_id']
    operations = export_operations(report, pid, metadata['n_nodes'], mount)
    operations.extend(generate_op_metadata(pid, metadata['n_nodes']))
    operations.append(
        {"name": "Darshan_start", "ph": "i", "ts": metadata['start_ts'] * 1e6, "pid": 0, "tid": 0, "s": "g"})
    operations.append(
        {"name": "Darshan_end", "ph": "i", "ts": metadata['end_ts'] * 1e6, "pid": 0, "tid": 0, "s": "g"})
    perfetto_trace = {
        "traceEvents": operations,
        "metadata": metadata,
    }
    with gzip.open(os.path.join(output_directory, trace.split('/')[-1] + '.json.gz'), 'wt') as f:
        json.dump(perfetto_trace, f, indent=2)
    return ''


def generate_traces_from_directory(darshan_directory: str, output_directory: str, mount: str = '/'):
    traces_to_convert = []
    for file in os.listdir(darshan_directory):
        if file.endswith('.darshan'):
            traces_to_convert.append(file)
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    process_pool = ProcessPool(os.cpu_count() - 1)
    for trace in traces_to_convert:
        process_pool.submit(generate_trace_event_json, os.path.join(darshan_directory, trace), output_directory, mount)
    process_pool.wait_completion()
    for result in process_pool.get_result():
        if result.startswith('error'):
            with open(os.path.join(output_directory, 'errors.txt'), 'a') as f:
                f.write(result + '\n')