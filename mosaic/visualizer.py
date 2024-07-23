import os
from datetime import timedelta, datetime

import plotly.graph_objs as go
from plotly.offline import plot


def visualize(trace_write: dict, pattern_write: list, write_classes: list, trace_read: dict, pattern_read: list,
              read_classes: list, output_dir: str, mount: str) -> None:
    """
    Generate html plots to visualize a trace
    @param trace_write: dictionary containing write operations
    @param pattern_write: list containing write segments
    @param write_classes: list containing write classes
    @param trace_read: dictionary containing read operations
    @param pattern_read: list containing read segments
    @param read_classes: list containing read classes
    @param output_dir: directory in which to save html file
    @param mount: PFS mounting point
    """
    job_scatter_write = create_job_trace(trace_write, 'write', mount)
    classification_scatter_write = create_categorizer_trace(pattern_write, trace_write['infos']['start_ts'],
                                                            trace_write['infos']['end_ts'], 'write')
    layout_write = go.Layout(title=f"Write Operations ({', '.join(write_classes)})",
                             xaxis=dict(title='Timestamp'),
                             yaxis=dict(title='Ranks Writing'))
    fig_write = go.Figure(data=[job_scatter_write, classification_scatter_write], layout=layout_write)
    job_scatter_read = create_job_trace(trace_read, 'read', mount)
    classification_scatter_read = create_categorizer_trace(pattern_read, trace_write['infos']['start_ts'],
                                                           trace_write['infos']['end_ts'], 'read')
    layout_read = go.Layout(title=f"Read Operations ({', '.join(read_classes)})",
                            xaxis=dict(title='Timestamp'),
                            yaxis=dict(title='Ranks Reading'))
    fig_read = go.Figure(data=[job_scatter_read, classification_scatter_read], layout=layout_read)
    plot1_html, plot2_html = None, None
    if len(pattern_write) > 0:
        plot1_html = plot(fig_write, output_type='div', include_plotlyjs=True)
    if len(pattern_read) > 0:
        plot2_html = plot(fig_read, output_type='div', include_plotlyjs=len(pattern_write) == 0)
    with open(os.path.join(output_dir, trace_write['infos']['file'] + '.html'), 'w') as f:
        if plot1_html:
            f.write(plot1_html)
        if plot2_html:
            f.write(plot2_html)


def create_job_trace(job: dict, operation_type: str, mount: str) -> go.scatter:
    """
    Create a scatter plot from a job original trace
    @param job: dictionary containing original operations
    @param operation_type: type of operation (read/write)
    @param mount: PFS mounting point
    @return: scatter plot of activity contained in trace
    """
    accesses = list(
        filter(lambda x: x['written' if operation_type == 'write' else 'read'] != 0 and x['file'].startswith(mount),
               job['access']))
    access_next_starts = sorted(accesses, key=lambda x: x[f'{operation_type}_start_ts'])
    access_next_ends = sorted(accesses, key=lambda x: x[f'{operation_type}_end_ts'])

    x_ts, y_amount = [], []

    x_ts.append(job['infos']['start_ts'])
    y_amount.append(0)

    while access_next_ends:
        next_start = access_next_starts[0] if access_next_starts else None
        next_end = access_next_ends[0]
        if access_next_starts and next_start[f'{operation_type}_start_ts'] < next_end[f'{operation_type}_end_ts']:
            earliest_operation = next_start
            earliest_timestamp = next_start[f'{operation_type}_start_ts']
            operation = int.__add__
            access_next_starts.pop(0)
        else:
            earliest_operation = next_end
            earliest_timestamp = next_end[f'{operation_type}_end_ts']
            operation = int.__sub__
            access_next_ends.pop(0)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        y_amount.append(y_amount[-1])
        y_amount.append(operation(y_amount[-1], (job['infos']['nprocs'] if earliest_operation['rank'] == -1 else 1)))

    x_ts.append(job['infos']['end_ts'])
    y_amount.append(0)

    return go.Scatter(x=x_ts, y=y_amount, mode='lines', name=f'{operation_type.capitalize()} Operations',
                      line=dict(color='red' if operation_type == 'write' else 'blue', width=3))


def create_categorizer_trace(segments: list, start: datetime, end: datetime, operation_type: str) -> go.Scatter:
    """
    Create a scatter plot from detected periodic operations
    @param segments: list of generated segments
    @param start: start timestamp
    @param end: end to timestamp
    @param operation_type: type of operation (read/write)
    @return: scatter plot of generated segments
    """
    x_ts, y_amount = [], []

    x_ts.append(start)
    y_amount.append(0)

    for segment in segments:
        for i in range(segment['segments_cnt']):
            x_ts.append(segment['start_ts'] + timedelta(seconds=i * segment['duration_avg']))
            x_ts.append(segment['start_ts'] + timedelta(seconds=i * segment['duration_avg']))
            y_amount.append(y_amount[-1])
            y_amount.append(y_amount[-1] + segment['n_ranks_avg'])
            x_ts.append(
                segment['start_ts'] + timedelta(seconds=i * segment['duration_avg'] + segment['working_time_avg']))
            x_ts.append(
                segment['start_ts'] + timedelta(seconds=i * segment['duration_avg'] + segment['working_time_avg']))
            y_amount.append(y_amount[-1])
            y_amount.append(y_amount[-1] - segment['n_ranks_avg'])

    x_ts.append(end)
    y_amount.append(0)

    return go.Scatter(x=x_ts, y=y_amount, mode='lines', name=f'{operation_type.capitalize()} Operations',
                      line=dict(color='blue' if operation_type == 'write' else 'red', width=3))


def create_metadata_trace(trace: dict, operation_type: str, mount: str) -> go.Scatter:
    """
    Create a scatter plot representing metadata activity
    @param trace: dictionary representation of the trace
    @param operation_type: type of operation (read/write)
    @param mount: PFS mounting point
    @return: scatter plot of metadata requests
    """
    accesses = list(
        filter(lambda x: x['written' if operation_type == 'write' else 'read'] != 0 and x['file'].startswith(mount),
               trace['access']))
    access_next_starts = sorted(accesses, key=lambda x: x[f'{operation_type}_start_ts'])
    access_next_ends = sorted(accesses, key=lambda x: x[f'{operation_type}_end_ts'])

    x_ts, y_amount = [], []

    x_ts.append(trace['infos']['start_ts'])
    y_amount.append(0)

    while access_next_ends:
        next_start = access_next_starts[0] if access_next_starts else None
        next_end = access_next_ends[0]
        if access_next_starts and next_start[f'{operation_type}_start_ts'] < next_end[f'{operation_type}_end_ts']:
            earliest_timestamp = next_start[f'{operation_type}_start_ts']
            operation = next_start['seeks'] + next_start['opens']
            access_next_starts.pop(0)
        else:
            earliest_timestamp = next_end[f'{operation_type}_end_ts']
            operation = next_end['opens']
            access_next_ends.pop(0)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        y_amount.append(0)
        y_amount.append(operation)
        y_amount.append(0)

    x_ts.append(trace['infos']['end_ts'])
    y_amount.append(0)

    return go.Scatter(x=x_ts, y=y_amount, mode='lines', name='Metadata Operations',
                      line=dict(color='green', width=3))
