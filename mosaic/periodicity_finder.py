import os
import statistics
from collections import Counter
from copy import copy

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


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


def segment_characterization(operations_per_segment: dict, segments: list) -> dict:
    """
    Characterize a group of segments
    @param operations_per_segment: dictionary of operations contained in each segment
    @param segments: list of segments in a group
    @return: dictionary with metrics about characterized segment group
    """
    start_ts = segments[0][0]
    end_ts = segments[-1][1] if len(segments) > 1 else max(
        map(lambda op: op['ts'] + op['dur'], operations_per_segment[segments[0]]))

    working_times, n_ranks = compute_activity_stats(segments, operations_per_segment)

    stats = {
        'start_ts': start_ts,
        'end_ts': end_ts,
        'segments_cnt': len(segments),
        'n_ranks_avg': statistics.mean(n_ranks),
        'duration_avg': (segments[-1][0] - start_ts) / (len(segments) - 1) if len(segments) > 1 else end_ts - start_ts,
        'working_time_avg': statistics.mean(working_times),
        'working_time_cv': statistics.stdev(working_times) / statistics.mean(working_times) if len(
            working_times) > 2 else 0,
        'data_operated_avg': statistics.mean(
            map(lambda op_l: sum(map(lambda op: op['args']['count'], op_l)), operations_per_segment.values())),
        'metadata_operations_avg': statistics.mean(
            map(lambda op_l: sum(map(lambda op: 2 * op['args']['opens'], op_l)),
                operations_per_segment.values())) + statistics.mean(
            map(lambda op_l: sum(map(lambda op: op['args']['seeks'], op_l)),
                operations_per_segment.values())),
    }
    return stats


def compute_activity_stats(segments: list, operations_per_segment: dict) -> (list, list):
    """
    Compute activity stats for a group of segments
    @param segments: list of segments in a group
    @param operations_per_segment: dictionary of operations contained in each segment
    @return: list of activity ratio per segment, list of average number of ranks per segment
    """
    working_times = []
    n_ranks = []
    for segment in segments:
        s = 0
        earliest_start = None
        latest_end = None
        for operation in operations_per_segment[segment]:
            # TODO change to real I/O time if possible, instead of full operation duration
            s += operation['dur']
            if not earliest_start:
                earliest_start = operation['ts']
            else:
                earliest_start = min(operation['ts'], earliest_start)
            if not latest_end:
                latest_end = operation['ts'] + operation['dur']
            else:
                latest_end = max(operation['ts'] + operation['dur'], latest_end)
        working_times.append(latest_end - earliest_start)
        n_ranks.append(s / working_times[-1])
    return working_times, n_ranks


def remove_characterized_segments(segments: list, start: float, end: float) -> list:
    """
    Remove a segment
    @param segments: list of all segments
    @param start: start timestamp of segment to remove
    @param end: stop timestamp of segment to remove
    @return: list of segments without the one removed
    """
    return list(filter(lambda s: s[0] > end or s[1] < start, segments))


def merge_neighbours(operations: list, total_seconds: int, avg_empty: float) -> None:
    """
    Merge neighboring operations
    @param operations: list of all operations
    @param total_seconds: trace's duration in seconds
    @param avg_empty: average duration between two operations
    """
    n_ops = len(operations)
    i = 0
    while i < n_ops - 1:
        o1_s, o1_e = operations[i]['ts'], operations[i]['ts'] + operations[i]['dur']
        o2_s, o2_e = operations[i + 1]['ts'], operations[i + 1]['ts'] + operations[i + 1]['dur']
        d = o2_s - o1_e
        dt = o2_e - o1_s
        if (d < .001 * total_seconds or d < 0.75 * avg_empty or d / dt < .01) and o2_s - o1_e < 1.5 * max(o1_e - o1_s,
                                                                                                          o2_e - o2_s):
            operations[i] = new_operation_from_merge(operations, i)
            operations.pop(i + 1)
            n_ops -= 1
        else:
            i += 1


def new_operation_from_merge(operations: list, i: int) -> dict:
    """
    Merge two operations to create a new one
    @param operations: list of all operations
    @param i: index of the first operation to merge
    @return: dictionary representation of the merged operation
    """
    new_op = copy(operations[i])
    new_op['dur'] = operations[i + 1]['ts'] + operations[i + 1]['dur'] - new_op['ts']
    new_op['args']['count'] += operations[i + 1]['args']['count']
    new_op['args']['speed'] = new_op['args']['count'] / new_op['dur']
    if new_op['args']['file'] != operations[i + 1]['args']['file']:
        new_op['args']['file'] = os.path.commonprefix([new_op['args']['file'], operations[i + 1]['args']['file']])
    if new_op['args']['opens'] is not None and operations[i + 1]['args']['opens'] is not None:
        new_op['args']['opens'] += operations[i + 1]['args']['opens']
    else:
        new_op['args']['opens'] = None
    if new_op['args']['seeks'] is not None and operations[i + 1]['args']['seeks'] is not None:
        new_op['args']['seeks'] += operations[i + 1]['args']['seeks']
    else:
        new_op['args']['seeks'] = None
    if new_op['args']['offset'] != operations[i + 1]['args']['offset']:
        new_op['args']['offset'] = None
    return new_op


def compute_metadata_stats(trace: dict, mount: str, spike_threshold: int) -> dict:
    """
    Compute metadata statistics for a trace
    @param trace: dictionary representation of the trace
    @param mount: PFS mounting point
    @param spike_threshold: amount of metadata requests per second from which they are considered impactful
    @return: dictionary containing metadata statistics
    """
    windows = {}
    operations = list(filter(lambda x: x['name'] in ['read', 'write'] and x['args']['opens'] is not None and x['args'][
        'seeks'] is not None and x['args']['file'].startswith(mount), trace['traceEvents']))
    for operation in operations:
        if operation['args']['opens'] + operation['args']['seeks'] == 0:
            continue
        timestamp_start = operation['ts']
        timestamp_end = timestamp_start + operation['dur']
        if timestamp_start in windows:
            windows[timestamp_start] += operation['args']['opens'] + operation['args']['seeks']
        else:
            windows[timestamp_start] = operation['args']['opens'] + operation['args']['seeks']
        if timestamp_end in windows:
            windows[timestamp_end] += operation['args']['opens']
        else:
            windows[timestamp_end] = operation['args']['opens']
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
        filter(lambda x: x['name'] == operation_type and x['args']['file'].startswith(mount),
               trace['traceEvents'])), key=lambda x: x['ts'])

    for op in operations:
        op['ts'] *= 1e-6
        op['dur'] *= 1e-6

    total_amount = sum(map(lambda x: x['args']['count'], operations))

    if total_amount == 0:
        return [], trace

    empty_count, total_empty_duration = compute_inactivity_stats(operations)

    if empty_count > 0:
        merge_neighbours(operations, trace['metadata']['end_ts'] - trace['metadata']['start_ts'],
                         total_empty_duration / empty_count)

    segments, operations_per_segments = create_segments(operations)

    classified_segments = []
    while segments:
        classified_segments += classify_one_segment_group(segments, operations_per_segments)
        operations_per_segments = {key: value for key, value in operations_per_segments.items() if key in segments}

    return sorted(classified_segments, key=lambda x: x['start_ts']), trace


def compute_inactivity_stats(operations: list) -> (int, int):
    """
    Compute inactivity stats
    @param operations: list of all operations
    @return: number of inactive segments, total inactivity duration in seconds
    """
    empty_count = 0
    total_empty_duration = 0
    for i in range(len(operations) - 1):
        if operations[i + 1]['ts'] > operations[i]['ts'] + operations[i]['dur']:
            empty_count += 1
            total_empty_duration += operations[i + 1]['ts'] - operations[i]['ts'] + operations[i]['dur']
    return empty_count, total_empty_duration


def create_segments(operations: list) -> (list, dict):
    """
    Create segments from the list of operations
    @param operations: list of all operations
    @return: list of segments, dictionary of operations contained per segment
    """
    segments = []
    operations_per_segments = {}
    while operations:
        operations_in_segment, seg_start, latest_end = create_one_segment(operations)
        seg_end = latest_end
        # if last segment and amount close to previous one, expand current segment to match the length of the
        # previous one to potentially include it in periodic segments
        if not operations and segments:
            po_s, po_e = segments[-1]
            seg_end = seg_start + po_e - po_s
        segments.append((seg_start, seg_end))
        operations_per_segments[(seg_start, seg_end)] = operations_in_segment
    return segments, operations_per_segments


def create_one_segment(operations) -> (list, float, float):
    """
    Create one segment from the first operation in the list
    @param operations: list of remaining operations to include in segments
    @return: list of operations contained in the segment, start and end timestamps of segment
    """
    operations_in_segment = []
    new_op = copy(operations.pop(0))
    seg_start = new_op['ts']
    operations_in_segment.append(new_op)
    latest_end = new_op['ts'] + new_op['dur']
    while operations:
        next_op = operations[0]
        # the next operation is outside of this segment, break
        if next_op['ts'] > latest_end:
            latest_end = next_op['ts']
            break
        # the next operation is contained by this segment (start before end of operations in segment)
        else:
            latest_end = max(latest_end, next_op['ts'] + next_op['dur'])
            operations_in_segment.append(next_op)
            operations.pop(0)
    return operations_in_segment, seg_start, latest_end


def classify_one_segment_group(segments: list, operations_per_segments: dict) -> list:
    """
    Clusterize segments, characterize the smallest group, and remove members from remaining segments
    @param segments: list of segments
    @param operations_per_segments: dictionary of operations contained per segment
    @return: list of classified segments
    """
    classified_segments = []
    segment_durations = list(map(lambda s: s[1] - s[0], segments))
    segments_amount = np.array(
        [sum(map(lambda v: v['args']['count'], value)) for key, value in operations_per_segments.items()],
        dtype=np.float64)
    segment_duration_classes = clusterize(segment_durations, segments_amount)
    least_common_class, _ = Counter(segment_duration_classes).most_common()[-1]
    cleaned_segments, cleaned_labels, cleaned_operations_per_segment = copy(segments), copy(
        segment_duration_classes), copy(operations_per_segments)
    segment_group = get_segments_of_label(cleaned_labels, cleaned_segments, least_common_class)
    filtered_operation_per_segment = {key: value for key, value in cleaned_operations_per_segment.items() if
                                      key in segment_group}
    classified_segments.append(
        segment_characterization(filtered_operation_per_segment, segment_group))
    for seg in segment_group:
        if seg not in segments:
            segments = remove_characterized_segments(segments, seg[0], seg[1])
        else:
            segments.remove(seg)
    return classified_segments
