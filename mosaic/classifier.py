from datetime import timedelta, datetime

import numpy as np


def classify_trace(trace_operations: dict, significant_read: bool, significant_write: bool) -> dict:
    """
    Classify one trace
    @param trace_operations: dictionary containing all operations with periodicity
    @param significant_read: true if read operations are significant, false otherwise
    @param significant_write: true if write operations are significant, false otherwise
    @return: dictionary containing classes assigned
    """
    metadata_classes = classify_metadata(trace_operations['metadata'])
    read_classes = classify_accesses(trace_operations, 'read') if significant_read else ['read_insignificant']
    write_classes = classify_accesses(trace_operations, 'write') if significant_write else ['write_insignificant']
    return {
        'metadata_classes': metadata_classes,
        'read_classes': read_classes,
        'write_classes': write_classes,
    }


def classify_accesses(operations: dict, operation_type: str) -> list:
    """
    Classify operations of a given type (read/write)
    @param operations: dictionary containing all operations with periodicity
    @param operation_type: type of operation to classify
    @return: list of classes assigned
    """
    classes = []
    classes += classify_access_temporality(operations, operation_type)
    classes += classify_periodicity(operations, operation_type)
    return classes


def classify_access_temporality(patterns: dict, operation_type: str) -> list:
    """
    Classify access temporality of operations of given type (read/write)
    @param patterns: dictionary containing all operations with periodicity
    @param operation_type: type of operation to classify
    @return: list of temporality classes assigned
    """
    classes = []

    histogram, hist_cv, hist_cv_before_end, hist_cv_after_start, hist_cv_start_end = compute_amount_histogram(
        load_operations(patterns[operation_type]),
        patterns['infos'][
            'start_ts'],
        patterns['infos']['end_ts'])

    if histogram[0] > 2 * (histogram[1] + histogram[2]):
        classes.append(f'{operation_type}_on_start')
    elif histogram[0] < (histogram[1] + histogram[2]) / 5 and hist_cv_after_start < .25:
        classes.append(f'{operation_type}_after_start')
    if histogram[2] > 2 * (histogram[0] + histogram[1]):
        classes.append(f'{operation_type}_on_end')
    elif histogram[2] < (histogram[0] + histogram[1]) / 5 and hist_cv_before_end < .25:
        classes.append(f'{operation_type}_before_end')
    if histogram[0] + histogram[2] > 4 * histogram[1] and hist_cv_start_end < .25:
        classes.append(f'{operation_type}_on_start_and_on_end')
    if histogram[1] > 4 * (histogram[0] + histogram[2]):
        classes.append(f'{operation_type}_after_start_before_end')
    if hist_cv < .25:
        classes.append(f'{operation_type}_steady')

    if not classes:
        classes.append(f'{operation_type}_unclear_pattern')

    return classes


def classify_periodicity(patterns: dict, operation_type: str) -> list:
    """
    Classify periodicity of operations of given type (read/write)
    @param patterns: dictionary containing all operations with periodicity
    @param operation_type: type of operation to classify
    @return: list of periodicity classes assigned
    """
    classes = []

    single_access_pattern_count = sum(
        map(lambda p: p['segments_cnt'], filter(lambda p: p['segments_cnt'] == 1, patterns[operation_type])))
    total_periodic_access_count = sum(
        map(lambda p: p['segments_cnt'], filter(lambda p: p['segments_cnt'] != 1, patterns[operation_type])))
    if total_periodic_access_count > 3 * single_access_pattern_count:
        classes.append(f'{operation_type}_periodic')

        mean_periodic_duration = ((max(
            map(lambda p: p['end_ts'],
                filter(lambda p: p['segments_cnt'] != 1, patterns[operation_type]))) - min(
            map(lambda p: p['start_ts'],
                filter(lambda p: p['segments_cnt'] != 1,
                       patterns[operation_type])))).total_seconds()) / total_periodic_access_count
        if mean_periodic_duration <= 30:
            classes.append(f'{operation_type}_periodic_s')
        elif mean_periodic_duration <= 1800:
            classes.append(f'{operation_type}_periodic_min')
        elif mean_periodic_duration <= 43200:
            classes.append(f'{operation_type}_periodic_h')
        else:
            classes.append(f'{operation_type}_periodic_day_or_more')

        mean_periodic_activity = sum(map(lambda p: p['segments_cnt'] * p['working_time_avg'] / p['duration_avg'],
                                         filter(lambda p: p['segments_cnt'] != 1,
                                                patterns[operation_type]))) / total_periodic_access_count

        if mean_periodic_activity <= .25:
            classes.append(f'{operation_type}_periodic_low_busy_time')
        elif mean_periodic_activity >= .75:
            classes.append(f'{operation_type}_periodic_high_busy_time')

    return classes


def classify_metadata(stats: dict) -> list:
    """
    Classify metadata impact of a trace
    @param stats: dictionary containing metadata metrics
    @return: list of metadata classes assigned
    """
    classes = []
    if stats['highest_spike'] > 250:
        classes.append('metadata_high_spike')
    if stats['operations_per_second'] > 50 and stats['spike_count'] > 5:
        classes.append('metadata_high_density')
    if stats['spike_count'] > 5:
        classes.append('metadata_multiple_spikes')
    if not classes:
        classes.append('metadata_insignificant_load')

    return classes


def load_operations(patterns: list) -> dict:
    """
    Load all operations in a dictionary with timestamps as keys
    @param patterns: list of operations
    @return: dictionary containing all operations with timestamps as keys
    """
    operations = {}
    for pattern in patterns:
        for i in range(pattern['segments_cnt']):
            start = pattern['start_ts'] + timedelta(seconds=i * pattern['duration_avg'])
            end = start + timedelta(seconds=pattern['working_time_avg'])
            if (start, end) not in operations:
                operations[(start, end)] = 0
            operations[(start, end)] += pattern['data_operated_avg']
    return operations


def compute_amount_histogram(operations: dict, start: datetime, end: datetime) -> (list, float, float, float, float):
    """
    Create an histogram of the volume of operations in 3 time chunks
    @param operations: dictionary containing all operations
    @param start: start timestamp
    @param end: end timestamp
    @return: list of quantity in all chunks, coefficient of variations between all chunks
    """
    amount_histogram = [0, 0, 0]
    start_end = start + timedelta(seconds=(end - start).total_seconds() / 4)
    end_start = end - timedelta(seconds=(end - start).total_seconds() / 4)
    for (op_start, op_end) in operations:
        amount = operations[(op_start, op_end)]
        duration = (op_end - op_start).total_seconds()
        s1_amount = max(0, min(1, (start_end - op_start).total_seconds() / duration)) * amount
        s3_amount = max(0, min(1, (op_end - end_start).total_seconds() / duration)) * amount
        s2_amount = amount - s1_amount - s3_amount
        amount_histogram[0] += s1_amount
        amount_histogram[1] += s2_amount
        amount_histogram[2] += s3_amount
    cv_total = (np.std(
        np.array(
            [amount_histogram[0], amount_histogram[1] / 2, amount_histogram[1] / 2, amount_histogram[2]]),
        dtype=np.float64) / np.mean(
        np.array(
            [amount_histogram[0], amount_histogram[1] / 2, amount_histogram[1] / 2, amount_histogram[2]]),
        dtype=np.float64)) if sum(amount_histogram) > 0 else 0
    cv_before_end = (np.std(
        np.array(
            [amount_histogram[0], amount_histogram[1] / 2, amount_histogram[1] / 2]),
        dtype=np.float64) / np.mean(
        np.array(
            [amount_histogram[0], amount_histogram[1] / 2, amount_histogram[1] / 2]),
        dtype=np.float64)) if sum(amount_histogram[:-1]) > 0 else 0
    cv_after_start = (np.std(
        np.array(
            [amount_histogram[1] / 2, amount_histogram[1] / 2, amount_histogram[2]]),
        dtype=np.float64) / np.mean(
        np.array(
            [amount_histogram[1] / 2, amount_histogram[1] / 2, amount_histogram[2]]),
        dtype=np.float64)) if sum(amount_histogram[1:]) > 0 else 0
    cv_start_and_end = (np.std(
        np.array(
            [amount_histogram[0], amount_histogram[2]]),
        dtype=np.float64) / np.mean(
        np.array(
            [amount_histogram[0], amount_histogram[2]]),
        dtype=np.float64)) if amount_histogram[0] + amount_histogram[2] > 0 else 0
    return amount_histogram, cv_total, cv_before_end, cv_after_start, cv_start_and_end
