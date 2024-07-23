import statistics
from collections import Counter
from copy import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from darshan.report import DarshanReport
from sklearn.cluster import MeanShift, estimate_bandwidth


def load_trace(report: DarshanReport, name: str, mount: str = '') -> dict:
    """
    Load a darshan trace in a dictionary with all information needed
    @param report: DarshanReport object
    @param name: trace name
    @param mount: PFS mounting point
    @return: dictionary containing all required information
    """
    trace = {'infos': get_job_infos(report, name)}
    if 'MPI-IO' in report.records.keys():
        module = 'MPIIO'
    elif 'POSIX' in report.records.keys():
        module = 'POSIX'
    elif 'STDIO' in report.records.keys():
        module = 'STDIO'
    else:
        raise RuntimeError(
            f'Application did not use MPI-IO, POSIX or STDIO, available modules: {", ".join(report.records.keys())}')

    module_df = pd.merge(report.records['MPI-IO' if module == 'MPIIO' else module].to_df()['counters'],
                         report.records['MPI-IO' if module == 'MPIIO' else module].to_df()['fcounters'],
                         left_on=['id', 'rank'], right_on=['id', 'rank'], how="inner", validate="many_to_many")

    if module == 'MPIIO':
        posix_df_c = report.records['POSIX'].to_df()['counters']
    else:
        posix_df_c = None

    # check if some columns contains negative numbers even if it should not be possible
    if module_df[[f'{module}_BYTES_READ', f'{module}_BYTES_WRITTEN']].lt(0).any().any():
        raise RuntimeError('Darshan trace is inconsistent (contains negative numbers where it should not).')

    trace['module'] = get_module_stats(module_df, module, report)
    trace['access'] = []

    for index in module_df.index:
        if report.name_records[module_df['id'][index]].startswith(mount):
            trace['access'].append(
                parse_access(module_df, module, index, posix_df_c, report.name_records[module_df['id'][index]],
                             trace['infos']['nprocs'], trace['infos']['start_ts']))

    trace['access'] = sort_accesses(trace['access'])

    def get_end_ts(operation: dict) -> datetime:
        if operation['read'] == 0:
            return operation['write_end_ts']
        if operation['written'] == 0:
            return operation['read_end_ts']
        return max(operation['read_end_ts'], operation['write_end_ts'])

    trace['infos']['end_ts'] = max(trace['infos']['end_ts'],
                                 max(map(get_end_ts, trace['access']), default=trace['infos']['end_ts']))

    return trace


def get_job_infos(report: DarshanReport, name: str) -> dict:
    """
    Load job information from trace
    @param report: report object
    @param name: trace name
    @return: dictionary of job information
    """
    return {
        'file': name,
        'uid': report.metadata['job']['uid'],
        'job_id': report.metadata['job']['jobid'],
        'start_ts': datetime.fromtimestamp(
            report.metadata['job']['start_time_sec'] + report.metadata['job']['start_time_nsec'] / 1e9),
        'end_ts': datetime.fromtimestamp(
            report.metadata['job']['end_time_sec'] + report.metadata['job']['end_time_nsec'] / 1e9),
        'run_time': report.metadata['job']['run_time'],
        'exe': report.metadata['exe'],
        'nprocs': report.metadata['job']['nprocs']
    }


def get_module_stats(module_df: pd.DataFrame, module: str, report: DarshanReport) -> dict:
    """
    Load module global information
    @param module_df: dataframe representation of the module
    @param module: module name
    @param report: report object
    @return: dictionary of module information
    """
    return {
        'read': module_df[f'{module}_BYTES_READ'].sum(),
        'written': module_df[f'{module}_BYTES_WRITTEN'].sum(),
        'opens': module_df[f'{module}_OPENS'].sum() if module != 'MPIIO' else
        report.records['POSIX'].to_df()['counters']['POSIX_OPENS'].sum(),
        'seeks': module_df[f'{module}_SEEKS'].sum() if module != 'MPIIO' else
        report.records['POSIX'].to_df()['counters']['POSIX_SEEKS'].sum(),
        'read_process_count': report.metadata['job']['nprocs'] if -1 in pd.unique(
            (module_df.loc[module_df[f'{module}_BYTES_READ'] != 0])['rank']) else len(
            pd.unique((module_df.loc[module_df[f'{module}_BYTES_READ'] != 0])['rank'])),
        'write_process_count': report.metadata['job']['nprocs'] if -1 in pd.unique(
            (module_df.loc[module_df[f'{module}_BYTES_WRITTEN'] != 0])['rank']) else len(
            pd.unique((module_df.loc[module_df[f'{module}_BYTES_WRITTEN'] != 0])['rank']))
    }


def parse_access(module_df: pd.DataFrame, module_name: str, index: int, posix_df: pd.DataFrame, file: str, nprocs: int,
                 start_ts: datetime) -> dict:
    """
    Parse an operation from module dataframe
    @param module_df: module dataframe
    @param module_name: module name
    @param index: operation index in DF
    @param posix_df: posix dataframe for metadata information
    @param file: accessed file
    @param nprocs: number of job's processors
    @param start_ts: job's start time
    @return: dictionary representation of the operation
    """
    return {
        'id': module_df['id'][index],
        'file': file,
        'rank': module_df['rank'][index],
        'opens': module_df[f'{module_name}_OPENS'][index] if module_name != 'MPIIO' else posix_df['POSIX_OPENS'][index],
        'seeks': module_df[f'{module_name}_SEEKS'][index] if module_name != 'MPIIO' else posix_df['POSIX_SEEKS'][index],
        'read_total_duration': (nprocs if module_df['rank'][index] == -1 else 1) * (
                module_df[f'{module_name}_F_READ_END_TIMESTAMP'][index] -
                module_df[f'{module_name}_F_READ_START_TIMESTAMP'][index]),
        'write_total_duration': (nprocs if module_df['rank'][index] == -1 else 1) * (
                module_df[f'{module_name}_F_WRITE_END_TIMESTAMP'][index] -
                module_df[f'{module_name}_F_WRITE_START_TIMESTAMP'][index]),
        'read': module_df[f'{module_name}_BYTES_READ'][index],
        'read_duration': module_df[f'{module_name}_F_READ_END_TIMESTAMP'][index] -
                         module_df[f'{module_name}_F_READ_START_TIMESTAMP'][index],
        'read_speed': 0 if module_df[f'{module_name}_BYTES_READ'][index] == 0 else module_df[f'{module_name}_BYTES_READ'][
                                                                                  index] / (module_df[
                                                                                                f'{module_name}_F_READ_END_TIMESTAMP'][
                                                                                                index] - module_df[
                                                                                                f'{module_name}_F_READ_START_TIMESTAMP'][
                                                                                                index]),
        'read_start_ts': start_ts + timedelta(
            seconds=module_df[f'{module_name}_F_READ_START_TIMESTAMP'][index]) if module_df[f'{module_name}_BYTES_READ'][
                                                                                 index] != 0 else 0,
        'read_end_ts': start_ts + timedelta(
            seconds=module_df[f'{module_name}_F_READ_END_TIMESTAMP'][index]) if module_df[f'{module_name}_BYTES_READ'][
                                                                               index] != 0 else 0,
        'written': module_df[f'{module_name}_BYTES_WRITTEN'][index],
        'write_duration': module_df[f'{module_name}_F_WRITE_END_TIMESTAMP'][index] -
                          module_df[f'{module_name}_F_WRITE_START_TIMESTAMP'][index],
        'write_speed': 0 if module_df[f'{module_name}_BYTES_WRITTEN'][index] == 0 else
        module_df[f'{module_name}_BYTES_WRITTEN'][index] / (module_df[f'{module_name}_F_WRITE_END_TIMESTAMP'][index] -
                                                            module_df[f'{module_name}_F_WRITE_START_TIMESTAMP'][index]),
        'write_start_ts': start_ts + timedelta(
            seconds=module_df[f'{module_name}_F_WRITE_START_TIMESTAMP'][index]) if module_df[f'{module_name}_BYTES_WRITTEN'][
                                                                                  index] != 0 else 0,
        'write_end_ts': start_ts + timedelta(
            seconds=module_df[f'{module_name}_F_WRITE_END_TIMESTAMP'][index]) if module_df[f'{module_name}_BYTES_WRITTEN'][
                                                                                index] != 0 else 0,
    }


def sort_accesses(accesses: list) -> list:
    """
    Sort accesses chronologically
    @param accesses: list of all accesses
    @return: chronologically ordered list of accesses
    """
    def get_start_ts(operation: dict) -> datetime:
        if operation['read'] == 0:
            return operation['write_start_ts']
        if operation['written'] == 0:
            return operation['read_start_ts']
        return min(operation['read_start_ts'], operation['write_start_ts'])

    return sorted(filter(lambda x: x['read'] != 0 or x['written'] != 0, accesses), key=get_start_ts)


def clusterize(durations: list, amounts: np.array) -> list:
    """
    Split segments into groups based on their similarities
    @param durations: list of segments' durations
    @param amounts: list of segments' amounts
    @return: list of segments' group labels
    """
    if len(durations) < 2:
        return [0 for _ in range(len(durations))]
    cv1 = statistics.stdev(durations) / statistics.mean(durations)
    cv2 = statistics.stdev(amounts) / statistics.mean(amounts)
    mean1 = statistics.mean(durations)
    abs_diff1 = not any(map(lambda x: (abs(x - mean1) / mean1) > .25, durations))
    mean2 = statistics.mean(amounts)
    abs_diff2 = not any(map(lambda x: (abs(x - mean2) / mean2) > .25, amounts))
    same_amounts = cv2 < .01 and abs_diff2
    if cv1 < (.1 if same_amounts else .01) and cv2 < .005 and (same_amounts or (abs_diff1 and abs_diff2)):
        return [0 for _ in range(len(durations))]
    norm_data_dur = np.array(durations).reshape(-1, 1)
    norm_data_amo = np.array(amounts).reshape(-1, 1)
    if cv1 < (.1 if same_amounts else .01) and (same_amounts or abs_diff1):
        norm_data = norm_data_amo
    elif cv2 < .005 and abs_diff2:
        norm_data = norm_data_dur
    else:
        norm_data = np.concatenate((norm_data_dur, norm_data_amo), axis=1)
    mean_shift = MeanShift(bandwidth=max(estimate_bandwidth(norm_data, quantile=0.5), .1))
    mean_shift.fit(norm_data)
    return list(mean_shift.labels_)


def get_segments_of_label(labels: list, segments: list, target_label: int) -> list:
    """
    Only get the operations with a given label
    @param labels: list of labels
    @param segments: list of segments
    @param target_label: label to select
    @return: segments with target label
    """
    if len(labels) == 0:
        return []
    res = []
    while labels:
        label = labels.pop(0)
        seg = copy(segments.pop(0))
        if label == target_label:
            res.append(seg)
    return res


def segment_characterization(operations_per_segment: dict, segments: list, operation_type: str) -> dict:
    """
    Characterize a group of segments
    @param operations_per_segment: dictionary of operations contained in each segment
    @param segments: list of segments in a group
    @param operation_type: type of operation to characterize (read/write)
    @return: dictionary with metrics about characterised segment group
    """
    start_ts = segments[0][0]
    end_ts = segments[-1][1] if len(segments) > 1 else max(
        map(lambda op: op[f'{operation_type}_end_ts'], operations_per_segment[segments[0]]))

    working_times, n_ranks = compute_activity_stats(segments, operations_per_segment, operation_type)

    stats = {
        'start_ts': start_ts,
        'end_ts': end_ts,
        'segments_cnt': len(segments),
        'n_ranks_avg': statistics.mean(n_ranks),
        'duration_avg': (segments[-1][0] - start_ts).total_seconds() / (len(segments) - 1) if len(
            segments) > 1 else (end_ts - start_ts).total_seconds(),
        'working_time_avg': statistics.mean(working_times),
        'working_time_cv': statistics.stdev(working_times) / statistics.mean(working_times) if len(
            working_times) > 2 else 0,
        'data_operated_avg': statistics.mean(
            map(lambda op_l: sum(map(lambda op: op['written' if operation_type == 'write' else 'read'], op_l)),
                operations_per_segment.values())),
        'metadata_operations_avg': statistics.mean(
            map(lambda op_l: sum(map(lambda op: 2 * op['opens'], op_l)),
                operations_per_segment.values())) + statistics.mean(
            map(lambda op_l: sum(map(lambda op: op['seeks'], op_l)),
                operations_per_segment.values())),
    }
    return stats


def compute_activity_stats(segments: list, operations_per_segment: dict, operation_type: str) -> (list, list):
    """
    Compute activity stats for a group of segments
    @param segments: list of segments in a group
    @param operations_per_segment: dictionary of operations contained in each segment
    @param operation_type: type of operation to characterize (read/write)
    @return: list of activity ratio per segment, list of average number of ranks per segment
    """
    working_times = []
    n_ranks = []
    for segment in segments:
        s = 0
        earliest_start = None
        latest_end = None
        for operation in operations_per_segment[segment]:
            s += operation['read_total_duration' if operation_type == 'read' else 'write_total_duration']
            if not earliest_start:
                earliest_start = operation[f'{operation_type}_start_ts']
            else:
                earliest_start = min(operation[f'{operation_type}_start_ts'], earliest_start)
            if not latest_end:
                latest_end = operation[f'{operation_type}_end_ts']
            else:
                latest_end = max(operation[f'{operation_type}_end_ts'], latest_end)
        working_times.append((latest_end - earliest_start).total_seconds())
        n_ranks.append(s / working_times[-1])
    return working_times, n_ranks


def remove_characterized_segments(segments: list, start: datetime, end: datetime) -> list:
    """
    Remove a segment
    @param segments: list of all segments
    @param start: start timestamp of segment to remove
    @param end: stop timestamp of segment to remove
    @return: list of segments without the one removed
    """
    return list(filter(lambda s: s[0] > end or s[1] < start, segments))


def merge_neighbours(operations: list, operation_type: str, total_seconds: int, avg_empty: float) -> None:
    """
    Merge neighboring operations
    @param operations: list of all operations
    @param operation_type: type of operation to characterize (read/write)
    @param total_seconds: trace's duration in seconds
    @param avg_empty: average duration between two operations
    """
    n_ops = len(operations)
    i = 0
    while i < n_ops - 1:
        o1_s, o1_e = operations[i][f'{operation_type}_start_ts'], operations[i][f'{operation_type}_end_ts']
        o2_s, o2_e = operations[i + 1][f'{operation_type}_start_ts'], operations[i + 1][f'{operation_type}_end_ts']
        d = (o2_s - o1_e).total_seconds()
        dt = (o2_e - o1_s).total_seconds()
        if (d < .001 * total_seconds or d < 0.75 * avg_empty or d / dt < .01) and (
                o2_s - o1_e).total_seconds() < 1.5 * max((
                                                                 o1_e - o1_s).total_seconds(),
                                                         (
                                                                 o2_e - o2_s).total_seconds()):
            operations[i] = new_operation_from_merge(operations, i, operation_type)
            operations.pop(i + 1)
            n_ops -= 1
        else:
            i += 1


def new_operation_from_merge(operations: list, i: int, operation_type: str) -> dict:
    """
    Merge two operations to create a new one
    @param operations: list of all operations
    @param i: index of the first operation to merge
    @param operation_type: type of operation to characterize (read/write)
    @return: dictionary representation of the merged operation
    """
    o1_s, o1_e = operations[i][f'{operation_type}_start_ts'], operations[i][f'{operation_type}_end_ts']
    o2_s, o2_e = operations[i + 1][f'{operation_type}_start_ts'], operations[i + 1][f'{operation_type}_end_ts']
    return {
        'read': operations[i]['read'] + operations[i + 1]['read'] if operation_type == 'read' else None,
        'read_start_ts': min(o1_s, o2_s) if operation_type == 'read' else None,
        'read_end_ts': max(o1_e, o2_e) if operation_type == 'read' else None,
        'written': operations[i]['written'] + operations[i + 1][
            'written'] if operation_type == 'write' else None,
        'write_start_ts': min(o1_s, o2_s) if operation_type == 'write' else None,
        'write_end_ts': max(o1_e, o2_e) if operation_type == 'write' else None,
        'read_total_duration': operations[i]['read_total_duration'] + operations[i + 1]['read_total_duration'],
        'write_total_duration': operations[i]['write_total_duration'] + operations[i + 1][
            'write_total_duration'],
        'opens': operations[i]['opens'] + operations[i + 1]['opens'],
        'seeks': operations[i]['seeks'] + operations[i + 1]['seeks']
    }


def compute_metadata_stats(trace: dict, mount: str, spike_threshold: int) -> dict:
    """
    Compute metadata statistics for a trace
    @param trace: dictionary representation of the trace
    @param mount: PFS mounting point
    @param spike_threshold: amount of metadata requests per second from which they are considered impactful
    @return: dictionary containing metadata statistics
    """
    windows = {}
    operations = list(filter(lambda x: x['file'].startswith(mount), trace['access']))
    for operation in operations:
        if operation['opens'] + operation['seeks'] == 0:
            continue
        timestamp_start = min(int(operation['write_start_ts'].timestamp() if operation[
                                                                                 'write_start_ts'] != 0 else datetime.now().timestamp()),
                              int(operation['read_start_ts'].timestamp()) if operation[
                                                                                 'read_start_ts'] != 0 else datetime.now().timestamp())
        timestamp_end = max(int(operation['write_end_ts'].timestamp() if operation[
                                                                             'write_end_ts'] != 0 else datetime.strptime(
            '1980', '%Y').timestamp()),
                            int(operation['read_end_ts'].timestamp()) if operation[
                                                                             'read_end_ts'] != 0 else datetime.strptime(
                                '1980', '%Y').timestamp())
        if timestamp_start in windows:
            windows[timestamp_start] += operation['opens'] + operation['seeks']
        else:
            windows[timestamp_start] = operation['opens'] + operation['seeks']
        if timestamp_end in windows:
            windows[timestamp_end] += operation['opens']
        else:
            windows[timestamp_end] = operation['opens']
    if len(windows) == 0:
        return {
            'highest_spike': 0,
            'spike_count': 0,
            'average_per_spike': 0,
            'operations_per_second': 0,
            'operations_duration': 0
        }
    metadata_highest_spike = max(windows.values())
    metadata_spike_count = sum(val >= spike_threshold for val in windows.values())
    metadata_average = statistics.mean(windows.values())
    metadata_op_ps = (sum(windows.values()) / (max(windows.keys()) - min(windows.keys()))) if len(windows) > 1 else 0
    metadata_ops_duration = max(windows.keys()) - min(windows.keys())
    return {
        'highest_spike': metadata_highest_spike,
        'spike_count': metadata_spike_count,
        'average_per_spike': metadata_average,
        'operations_per_second': metadata_op_ps,
        'operations_duration': metadata_ops_duration
    }


def find_periodic_patterns(trace: dict, operation_type: str, mount: str) -> (list, dict):
    """
    Create and group segments
    @param trace: dictionary representation of the trace
    @param operation_type: type of operation to characterize (read/write)
    @param mount: PFS mounting point
    @return: list of dictionaries representing each group of segments
    """
    operations = sorted(list(
        filter(lambda x: x['written' if operation_type == 'write' else 'read'] != 0 and x['file'].startswith(mount),
               trace['access'])), key=lambda x: x[f'{operation_type}_start_ts'])

    total_amount = sum(map(lambda x: x['written' if operation_type == 'write' else 'read'], operations))
    opens = sum(map(lambda x: x['opens'], operations))
    seeks = sum(map(lambda x: x['seeks'], operations))

    if total_amount < 100e6 and opens <= trace['infos']['nprocs'] and seeks <= trace['infos']['nprocs']:
        return {}, trace

    empty_count, total_empty_duration = compute_inactivity_stats(operations, operation_type)

    if empty_count > 0:
        merge_neighbours(operations, operation_type,
                         (trace['infos']['end_ts'] - trace['infos']['start_ts']).total_seconds(),
                         total_empty_duration / empty_count)

    segments, operations_per_segments = create_segments(operations, operation_type)

    classified_segments = []
    while segments:
        classified_segments += classify_one_segment_group(segments, operation_type, operations_per_segments)
        operations_per_segments = {key: value for key, value in operations_per_segments.items() if key in segments}

    return sorted(classified_segments, key=lambda x: x['start_ts']), trace


def compute_inactivity_stats(operations: list, operation_type: str) -> (int, int):
    """
    Compute inactivity stats
    @param operations: list of all operations
    @param operation_type: type of operation to characterize (read/write)
    @return: number of inactive segments, total inactivity duration in seconds
    """
    empty_count = 0
    total_empty_duration = 0
    for i in range(len(operations) - 1):
        if operations[i + 1][f'{operation_type}_start_ts'] > operations[i][f'{operation_type}_end_ts']:
            empty_count += 1
            total_empty_duration += (operations[i + 1][f'{operation_type}_start_ts'] - operations[i][
                f'{operation_type}_end_ts']).total_seconds()
    return empty_count, total_empty_duration


def create_segments(operations: list, operation_type: str) -> (list, dict):
    """
    Create segments from the list of operations
    @param operations: list of all operations
    @param operation_type: type of operation to characterize (read/write)
    @return: list of segments, dictionary of operations contained per segment
    """
    segments = []
    operations_per_segments = {}
    while operations:
        operations_in_segment, seg_start, latest_end = create_one_segment(operations, operation_type)
        seg_end = latest_end
        # if last segment and amount close to previous one, expand current segment to match the length of the
        # previous one to potentially include it in periodic segments
        if not operations and segments:
            po_s, po_e = segments[-1]
            seg_end = seg_start + timedelta(seconds=(po_e - po_s).total_seconds())
        segments.append((seg_start, seg_end))
        operations_per_segments[(seg_start, seg_end)] = operations_in_segment
    return segments, operations_per_segments


def create_one_segment(operations, operation_type) -> (list, datetime, datetime):
    """
    Create one segment from the first operation in the list
    @param operations: list of remaining operations to include in segments
    @param operation_type: type of operation to characterize (read/write)
    @return: list of operations contained in the segment, start and end timestamps of segment
    """
    operations_in_segment = []
    new_op = copy(operations.pop(0))
    seg_start = new_op[f'{operation_type}_start_ts']
    operations_in_segment.append(new_op)
    latest_end = new_op[f'{operation_type}_end_ts']
    while operations:
        next_op = operations[0]
        # the next operation is outside of this segment, break
        if next_op[f'{operation_type}_start_ts'] > latest_end:
            latest_end = next_op[f'{operation_type}_start_ts']
            break
        # the next operation is contained by this segment (start before end of operations in segment)
        else:
            latest_end = max(latest_end, next_op[f'{operation_type}_end_ts'])
            operations_in_segment.append(next_op)
            operations.pop(0)
    return operations_in_segment, seg_start, latest_end


def classify_one_segment_group(segments: list, operation_type: str, operations_per_segments: dict) -> list:
    """
    Clusterize segments, characterize the smallest group, and remove members from remaining segments
    @param segments: list of segments
    @param operation_type: type of operation to characterize (read/write)
    @param operations_per_segments: dictionary of operations contained per segment
    @return: list of classified segments
    """
    classified_segments = []
    segment_durations = list(map(lambda s: (s[1] - s[0]).total_seconds(), segments))
    segments_amount = np.array(
        [sum(map(lambda v: v['written' if operation_type == 'write' else 'read'], value)) for key, value in
         operations_per_segments.items()], dtype=np.float64)
    segment_duration_classes = clusterize(segment_durations, segments_amount)
    least_common_class, _ = Counter(segment_duration_classes).most_common()[-1]
    cleaned_segments, cleaned_labels, cleaned_operations_per_segment = copy(segments), copy(
        segment_duration_classes), copy(operations_per_segments)
    segment_group = get_segments_of_label(cleaned_labels, cleaned_segments, least_common_class)
    filtered_operation_per_segment = {key: value for key, value in cleaned_operations_per_segment.items() if
                                      key in segment_group}
    classified_segments.append(
        segment_characterization(filtered_operation_per_segment, segment_group, operation_type))
    for seg in segment_group:
        if seg not in segments:
            segments = remove_characterized_segments(segments, seg[0], seg[1])
        else:
            segments.remove(seg)
    return classified_segments
