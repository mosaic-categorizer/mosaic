import math


def classify_trace(job: dict, trace_operations: dict) -> dict:
    """
    Classify one trace
    @param job: dictionary of the content of the trace
    @param trace_operations: dictionary containing all operations from the periodicity detection
    @return: dictionary containing classes assigned
    """
    classes = {
        "metadata_classes": classify_metadata(trace_operations["metadata"]),
        "read_classes": classify_accesses(job, trace_operations, "read"),
        "write_classes": classify_accesses(job, trace_operations, "write"),
    }
    return classes


def classify_accesses(job: dict, operations: dict, operation_type: str) -> list:
    """
    Classify operations of a given type (read/write)
    @param job: dictionary of the content of the trace
    @param operations: dictionary containing all operations from the periodicity detection
    @param operation_type: type of operation to classify
    @return: list of classes assigned
    """
    classes = [
        classify_access_duration(job, operation_type),
        classify_amounts(job, operation_type),
        classify_access_temporality(job, operation_type),
    ]
    classes += classify_periodicity(operations, operation_type)
    return classes


def classify_ftio(ftio_res: dict, operation_type: str) -> list:
    classes = []
    if operation_type in ftio_res.keys():
        classes.append(
            operation_type + get_periodicity_magnitude(ftio_res[operation_type])
        )
    return classes


def classify_access_duration(job: dict, operation_type: str) -> str:
    """
    Classify access duration of operations of given type (read/write)
    @param job: dictionary of the trace
    @param operation_type: type of operation to classify
    @return: duration class assigned
    """
    operations = [op for op in job["traceEvents"] if op["name"] == operation_type]
    total_io = sum(op["dur"] * 1e-6 for op in operations)
    if total_io == 0:
        return f"{operation_type}_duration_0"
    if total_io < 60:
        return f"{operation_type}_duration_less_1m"
    elif total_io < 300:
        return f"{operation_type}_duration_1m_to_5m"
    elif total_io < 1800:
        return f"{operation_type}_duration_5m_to_30m"
    elif total_io < 3600:
        return f"{operation_type}_duration_30m_to_1h"
    else:
        return f"{operation_type}_duration_1h_or_more"


def classify_access_temporality(job: dict, operation_type: str) -> str:
    """
    Classify access temporality of operations of given type (read/write)
    @param job: dictionary of the trace
    @param operation_type: type of operation to classify
    @return: list of temporality classes assigned
    """
    operations = [op for op in job["traceEvents"] if op["name"] == operation_type]
    active_chunks = find_active_chunks(
        operations, job["metadata"]["start_ts"], job["metadata"]["run_time"]
    )
    return f'{operation_type}_{"".join(["1" if active else "0" for active in active_chunks])}'


def classify_amounts(job: dict, operation_type: str) -> str:
    total_amount = sum(
        [
            op["args"]["count"]
            for op in job["traceEvents"]
            if op["name"] == operation_type
        ]
    )
    if total_amount == 0:
        return f"{operation_type}_0"
    if total_amount < 0.5 * 1e6:
        return f"{operation_type}_KB"
    if total_amount < 0.5 * 1e9:
        return f"{operation_type}_MB"
    if total_amount < 0.5 * 1e10:
        return f"{operation_type}_1_GB"
    if total_amount < 0.5 * 1e11:
        return f"{operation_type}_10_GB"
    if total_amount < 0.5 * 1e12:
        return f"{operation_type}_100_GB"
    if total_amount < 0.5 * 1e13:
        return f"{operation_type}_1_TB"
    return f"{operation_type}_10_TB_or_more"


def classify_periodicity(patterns: dict, operation_type: str) -> list:
    """
    Classify periodicity of operations of given type (read/write)
    @param patterns: dictionary containing all operations with periodicity
    @param operation_type: type of operation to classify
    @return: list of periodicity classes assigned
    """
    classes = []

    ftio_res = patterns["ftio"]
    classes += classify_ftio(ftio_res, operation_type)

    if patterns[operation_type]:
        total_periodic_access_count = sum(
            map(
                lambda p: p["segments_cnt"],
                filter(lambda p: p["segments_cnt"] != 1, patterns[operation_type]),
            )
        )
        if total_periodic_access_count > 2:
            mean_periodic_duration = (
                max(
                    map(
                        lambda p: p["end_ts"],
                        filter(
                            lambda p: p["segments_cnt"] != 1, patterns[operation_type]
                        ),
                    )
                )
                - min(
                    map(
                        lambda p: p["start_ts"],
                        filter(
                            lambda p: p["segments_cnt"] != 1, patterns[operation_type]
                        ),
                    )
                )
            ) / total_periodic_access_count
            classes.append(
                operation_type + get_periodicity_magnitude(mean_periodic_duration)
            )
    if len(classes) > 0:
        classes.append(f"{operation_type}_periodic")
    else:
        classes.append(f"{operation_type}_aperiodic")

    return classes


def get_periodicity_magnitude(mean_period) -> str:
    if mean_period <= 30:
        return "_periodic_s"
    elif mean_period <= 1800:
        return "_periodic_min"
    elif mean_period <= 43200:
        return "_periodic_h"
    else:
        return "_periodic_day_or_more"


def classify_metadata(stats: dict) -> list:
    """
    Classify metadata impact of a trace
    @param stats: dictionary containing metadata metrics
    @return: list of metadata classes assigned
    """
    classes = []
    if stats["highest_spike"] > 1000:
        classes.append("metadata_high_spike")
    if stats["operations_per_second"] > 50 and stats["spike_count"] > 5:
        classes.append("metadata_high_density")
    if stats["spike_count"] > 5:
        classes.append("metadata_multiple_spikes")
    if not classes:
        classes.append("metadata_insignificant_load")

    return classes


def load_operations(patterns: list) -> dict:
    """
    Load all operations in a dictionary with timestamps as keys
    @param patterns: list of operations
    @return: dictionary containing all operations with timestamps as keys
    """
    operations = {}
    for pattern in patterns:
        for i in range(pattern["segments_cnt"]):
            start = pattern["start_ts"] + (i * pattern["duration_avg"])
            end = start + pattern["working_time_avg"]
            if (start, end) not in operations:
                operations[(start, end)] = 0
            operations[(start, end)] += pattern["data_operated_avg"]
    return operations


def find_active_chunks(
    events: list, start: float, duration: float, n_chunks: int = 4
) -> list:
    """
    Create an histogram of the volume of operations in 3 time chunks
    @param events: list containing the operations from which the temporality is computed
    @param start: start timestamp
    @param duration: duration of the trace
    @param n_chunks: number of chunks to create, default is 4
    @return: list of booleans, True if activity in the chunk, False otherwise
    """
    active_chunks = [False for _ in range(n_chunks)]
    chunk_duration = duration / n_chunks
    for event in events:
        ts = event["ts"] * 1e-6
        dur = event["dur"] * 1e-6
        first_chunk = min(math.floor((ts - start) / chunk_duration), n_chunks - 1)
        last_chunk = min(
            math.ceil((ts + dur - start) / chunk_duration) - 1, n_chunks - 1
        )
        for i in range(first_chunk, last_chunk + 1):
            try:
                active_chunks[i] = True
            except IndexError:
                raise RuntimeError(
                    "Chunk index out of range when computing temporality, incoherent trace"
                )
    return active_chunks


def generate_trace_vector(trace_data: dict, merged_operations: dict) -> str:
    vect = ""
    vect += f'{trace_data["infos"]["run_time"]},{trace_data["module"]["read_duration"]},{trace_data["module"]["write_duration"]},{trace_data["module"]["read_process_count"]},{trace_data["module"]["write_process_count"]},{trace_data["module"]["read_operations"]},{trace_data["module"]["write_operations"]},{trace_data["module"]["read_files"]},{trace_data["module"]["written_files"]},{trace_data["module"]["read"]},{trace_data["module"]["written"]}'
    read_hist = find_active_chunks(
        load_operations(merged_operations["read"]),
        merged_operations["infos"]["start_ts"],
        merged_operations["infos"]["end_ts"],
    )[0]
    vect += f',{",".join(str(n / max(1, sum(read_hist))) for n in read_hist)}'
    write_hist = find_active_chunks(
        load_operations(merged_operations["write"]),
        merged_operations["infos"]["start_ts"],
        merged_operations["infos"]["end_ts"],
    )[0]
    vect += f',{",".join(str(n / max(1, sum(write_hist))) for n in write_hist)}'
    single_read_pattern_count = sum(
        map(
            lambda p: p["segments_cnt"],
            filter(lambda p: p["segments_cnt"] == 1, merged_operations["read"]),
        )
    )
    periodic_read_access_count = sum(
        map(
            lambda p: p["segments_cnt"],
            filter(lambda p: p["segments_cnt"] != 1, merged_operations["read"]),
        )
    )
    distinct_periodic_read_access_count = sum(
        p["segments_cnt"] != 1 for p in merged_operations["read"]
    )
    vect += f",{single_read_pattern_count},{periodic_read_access_count},{distinct_periodic_read_access_count}"
    single_write_pattern_count = sum(
        map(
            lambda p: p["segments_cnt"],
            filter(lambda p: p["segments_cnt"] == 1, merged_operations["write"]),
        )
    )
    periodic_write_access_count = sum(
        map(
            lambda p: p["segments_cnt"],
            filter(lambda p: p["segments_cnt"] != 1, merged_operations["write"]),
        )
    )
    distinct_periodic_write_access_count = sum(
        p["segments_cnt"] != 1 for p in merged_operations["write"]
    )
    vect += f",{single_write_pattern_count},{periodic_write_access_count},{distinct_periodic_write_access_count}"
    vect += f',{merged_operations["metadata"]["highest_spike"]},{merged_operations["metadata"]["spike_count"]},{merged_operations["metadata"]["operations_per_second"]}'
    return vect


def classify_file_temperatures(temps: dict, half_life: int = 60) -> str:
    total_temperature = 0
    for _, temp in temps.items():
        total_temperature += temp
    if total_temperature == 0:
        return f"temp_{half_life}s_frozen"
    if total_temperature < 10:
        return f"temp_{half_life}s_cold"
    if total_temperature < 100:
        return f"temp_{half_life}s_warm"
    if total_temperature < 500:
        return f"temp_{half_life}s_hot"
    return f"temp_{half_life}s_boiling"


def classify_multiple_temperatures(temps_mult_hf: dict) -> list:
    classes = []
    for half_life in temps_mult_hf.keys():
        classes.append(classify_file_temperatures(temps_mult_hf[half_life], half_life))
    return classes


def compute_file_temperatures(operations: dict, half_life: int = 60) -> dict:
    files = {}
    result = {}
    coeff = math.log(0.5) / half_life
    for operation in operations:
        if operation["name"] not in ["read", "write"]:
            continue
        file = operation["args"]["file"]
        if file not in files:
            files[file] = []
        files[file].append((operation["ts"] * 1e-6, operation["dur"] * 1e-6))
    for file in files:
        sorted_files = sorted(files[file], key=lambda x: x[0])
        score = 0
        latest_considered_op = 0
        for i in range(1, len(sorted_files)):
            if (
                sorted_files[latest_considered_op][0]
                + sorted_files[latest_considered_op][1]
                > sorted_files[i][0]
            ):
                continue
            relative_ts = (
                sorted_files[i][0]
                - sorted_files[latest_considered_op][0]
                - sorted_files[latest_considered_op][1]
            )
            score += math.exp(coeff * relative_ts)
            latest_considered_op = i
        result[file] = score
    return result


def compute_file_temperatures_multiple(operations: dict, half_lives: list) -> dict:
    temperatures = {}
    for half_life in half_lives:
        temperatures[half_life] = compute_file_temperatures(operations, half_life)
    return temperatures
