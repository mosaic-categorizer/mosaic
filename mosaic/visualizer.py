import csv
import json
import os
import re
from collections import defaultdict
from copy import copy
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from palettable.cartocolors.qualitative import Safe_10 as palette
import plotly.graph_objs as go
from plotly.offline import plot
import seaborn as sns

COLOR_MAPPING = {}


def visualize(
    job: dict,
    pattern_write: list,
    write_classes: list,
    pattern_read: list,
    read_classes: list,
    output_dir: str,
    mount: str,
) -> None:
    """
    Generate html plots to visualize a trace
    @param job: dictionary containing the job information
    @param pattern_write: list containing write segments
    @param write_classes: list containing write classes
    @param pattern_read: list containing read segments
    @param read_classes: list containing read classes
    @param output_dir: directory in which to save html file
    @param mount: PFS mounting point
    """
    job_scatter_write = create_job_trace(job, "write", mount)
    classification_scatter_write = create_categorizer_trace(
        pattern_write, job["metadata"]["start_ts"], job["metadata"]["end_ts"], "write"
    )
    layout_write = go.Layout(
        title=f"Write Operations ({', '.join(write_classes)})",
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Ranks Writing"),
    )
    fig_write = go.Figure(
        data=[job_scatter_write, classification_scatter_write], layout=layout_write
    )
    job_scatter_read = create_job_trace(job, "read", mount)
    classification_scatter_read = create_categorizer_trace(
        pattern_read, job["metadata"]["start_ts"], job["metadata"]["end_ts"], "read"
    )
    layout_read = go.Layout(
        title=f"Read Operations ({', '.join(read_classes)})",
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="Ranks Reading"),
    )
    fig_read = go.Figure(
        data=[job_scatter_read, classification_scatter_read], layout=layout_read
    )
    plot1_html, plot2_html = None, None
    if len(pattern_write) > 0:
        plot1_html = plot(fig_write, output_type="div", include_plotlyjs=False)
    if len(pattern_read) > 0:
        plot2_html = plot(fig_read, output_type="div", include_plotlyjs=False)
    with open(os.path.join(output_dir, job["metadata"]["file"] + ".html"), "w") as f:
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
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
        filter(
            lambda x: x["name"] == operation_type
            and x["args"]["file"].startswith(mount),
            job["traceEvents"],
        )
    )
    access_next_starts = sorted(accesses, key=lambda x: x["ts"])
    access_next_ends = sorted(accesses, key=lambda x: x["ts"] + x["dur"])

    x_ts, y_amount = [], []

    x_ts.append(job["metadata"]["start_ts"])
    y_amount.append(0)

    while access_next_ends:
        next_start = access_next_starts[0] if access_next_starts else None
        next_end = access_next_ends[0]
        if access_next_starts and next_start["ts"] < next_end["ts"] + next_end["dur"]:
            earliest_timestamp = next_start["ts"] * 1e-6
            operation = int.__add__
            access_next_starts.pop(0)
        else:
            earliest_timestamp = (next_end["ts"] + next_end["dur"]) * 1e-6
            operation = int.__sub__
            access_next_ends.pop(0)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        y_amount.append(y_amount[-1])
        y_amount.append(operation(y_amount[-1], 1))

    x_ts.append(job["metadata"]["end_ts"])
    y_amount.append(0)

    x_ts = [datetime.fromtimestamp(ts) for ts in x_ts]

    return go.Scatter(
        x=x_ts,
        y=y_amount,
        mode="lines",
        name=f"{operation_type.capitalize()} Operations",
        line=dict(color="red" if operation_type == "write" else "blue", width=3),
    )


def create_categorizer_trace(
    segments: list, start: float, end: float, operation_type: str
) -> go.Scatter:
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
        for i in range(segment["segments_cnt"]):
            x_ts.append(segment["start_ts"] + (i * segment["duration_avg"]))
            x_ts.append(segment["start_ts"] + (i * segment["duration_avg"]))
            y_amount.append(y_amount[-1])
            y_amount.append(y_amount[-1] + segment["n_ranks_avg"])
            x_ts.append(
                segment["start_ts"]
                + (i * segment["duration_avg"] + segment["working_time_avg"])
            )
            x_ts.append(
                segment["start_ts"]
                + (i * segment["duration_avg"] + segment["working_time_avg"])
            )
            y_amount.append(y_amount[-1])
            y_amount.append(y_amount[-1] - segment["n_ranks_avg"])

    x_ts.append(end)
    y_amount.append(0)

    x_ts = [datetime.fromtimestamp(ts) for ts in x_ts]

    return go.Scatter(
        x=x_ts,
        y=y_amount,
        mode="lines",
        name=f"{operation_type.capitalize()} Operations from Periodicity Detection",
        line=dict(color="blue" if operation_type == "write" else "red", width=3),
    )


def create_metadata_trace(trace: dict, operation_type: str, mount: str) -> go.Scatter:
    """
    Create a scatter plot representing metadata activity
    @param trace: dictionary representation of the trace
    @param operation_type: type of operation (read/write)
    @param mount: PFS mounting point
    @return: scatter plot of metadata requests
    """
    accesses = list(
        filter(
            lambda x: x["written" if operation_type == "write" else "read"] != 0
            and x["file"].startswith(mount),
            trace["access"],
        )
    )
    access_next_starts = sorted(accesses, key=lambda x: x[f"{operation_type}_start_ts"])
    access_next_ends = sorted(accesses, key=lambda x: x[f"{operation_type}_end_ts"])

    x_ts, y_amount = [], []

    x_ts.append(trace["infos"]["start_ts"])
    y_amount.append(0)

    while access_next_ends:
        next_start = access_next_starts[0] if access_next_starts else None
        next_end = access_next_ends[0]
        if (
            access_next_starts
            and next_start[f"{operation_type}_start_ts"]
            < next_end[f"{operation_type}_end_ts"]
        ):
            earliest_timestamp = next_start[f"{operation_type}_start_ts"]
            operation = next_start["seeks"] + next_start["opens"]
            access_next_starts.pop(0)
        else:
            earliest_timestamp = next_end[f"{operation_type}_end_ts"]
            operation = next_end["opens"]
            access_next_ends.pop(0)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        x_ts.append(earliest_timestamp)
        y_amount.append(0)
        y_amount.append(operation)
        y_amount.append(0)

    x_ts.append(trace["infos"]["end_ts"])
    y_amount.append(0)

    x_ts = [datetime.fromtimestamp(ts) for ts in x_ts]

    return go.Scatter(
        x=x_ts,
        y=y_amount,
        mode="lines",
        name="Metadata Operations",
        line=dict(color="green", width=3),
    )


def generate_class_repartition_wrt_io(
    sizes_per_class: dict, output_directory: str
) -> None:
    for class_ in sizes_per_class:
        p = sns.displot(sizes_per_class[class_], bins=100, kde=True, log_scale=True)
        p.savefig(os.path.join(output_directory, f"kde_{class_}.svg"))
        plt.close(p.figure)


def generate_box_plots(sizes_per_class: dict, output_directory: str) -> None:
    plots_fields = []
    if os.path.exists("box_plots.conf.csv"):
        with open("box_plots.conf.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                plots_fields.append(row)
        if set(sizes_per_class.keys()) - set([x for xs in plots_fields for x in xs]):
            print(
                f"Box plots generation - warning: classes {set(sizes_per_class.keys()) - set([x for xs in plots_fields for x in xs])} are not set to be exported in box plots in box_plots.conf.csv"
            )
    else:
        plots_fields = [sorted(sizes_per_class.keys())]
    for i in range(len(plots_fields)):
        fields = plots_fields[i]
        fig = go.Figure()
        for f in fields:
            if f not in sizes_per_class:
                continue
            violin = go.Violin(
                y=sizes_per_class[f],
                points=False,
                name=f,
                spanmode="hard",
                bandwidth=1e10,
            )
            fig.add_trace(violin)
        with open(
            os.path.join(
                output_directory, f"class_distribution_io_size_plot_{i + 1}.html"
            ),
            "w",
        ) as f:
            fig.update_traces(box_visible=True)
            fig.update_yaxes(type="log")
            f.write(plot(fig, output_type="div"))


def remove_svg_dimensions(filename) -> None:
    with open(filename, "r") as file:
        svg_content = file.read()
    svg_content = re.sub(r'width="\d+pt"', "", svg_content)
    svg_content = re.sub(r'height="\d+pt"', "", svg_content)
    with open(filename, "w") as file:
        file.write(svg_content)


def generate_donut_plot(
    title: str,
    plot_path: str,
    labels: list,
    values: list,
    inner_ring: list = None,
    ring_text: str = None,
) -> None:
    plt.figure(figsize=(6, 5))
    my_circle = plt.Circle((0, 0), 0.5, color="white")
    total = sum(values)
    labels_old = copy(labels)
    labels = [label.replace("\\n", "\n") for label in labels]
    legend_labels = [
        f"{label} ({pct:.1f}%)"
        for label, pct in zip(labels, [v / total * 100 for v in values])
    ]
    colors = [
        COLOR_MAPPING.get(get_label_discr(label), "#cccccc") for label in labels_old
    ]
    plt.pie(
        values,
        radius=1,
        labels=legend_labels,
        colors=colors,
        labeldistance=1.05,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    if inner_ring:
        plt.pie(
            inner_ring,
            radius=0.49,
            colors=["white", palette.hex_colors[len(COLOR_MAPPING)]],
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
        )
        p.gca().add_artist(plt.Circle((0, 0), 0.4, color="white"))
        plt.figtext(
            0.5,
            0.1,
            ring_text,
            ha="center",
            fontsize=8,
            color=palette.hex_colors[len(COLOR_MAPPING)],
        )
    plt.text(
        0,
        0,
        title,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(plot_path)
    remove_svg_dimensions(plot_path)
    plt.clf()


def generate_stacked_plot(plot_path: str, labels: list, values: list) -> None:
    fig, ax = plt.subplots(figsize=(4, 0.5))
    left = 0
    for i, (pct, label) in enumerate(zip(values, labels)):
        pct = pct * 100 / sum(values)
        ax.barh(
            0,
            pct,
            left=left,
            color=COLOR_MAPPING.get(get_label_discr(label), "#cccccc"),
            height=0.1,
        )
        if pct > 2.5:
            ax.text(
                left + pct / 2,
                0,
                f"{pct:.0f}%",
                va="center",
                ha="center",
                fontsize=4.5,
                color="white",
            )
        else:
            ax.text(
                left + pct / 2,
                0.05,
                f"{pct:.0f}%",
                va="bottom",
                ha="center",
                fontsize=4.5,
                color="black",
            )

        left += pct

    ax.set_ylim(-0.2, 0.2)
    ax.axis("off")

    short_labels = []
    for label in labels:
        short_labels.append(label.replace("read", "r").replace("write", "w"))
    ax.legend(
        short_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.7),
        ncol=len(labels),
        fontsize=5.5,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.2,
    )

    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
    remove_svg_dimensions(plot_path)
    plt.clf()


def filter_values(labels: list, values: list, threshold: float = -1) -> (list, list):
    ordered = sorted(list(zip(labels, values)), key=lambda x: x[1], reverse=True)
    sum_before_threshold = sum([val[1] for val in ordered])
    if threshold > -1:
        ordered = [
            val for val in ordered if (val[1] / sum_before_threshold) >= threshold
        ]
    filtered_labels, filtered_values = zip(*ordered)
    filtered_labels, filtered_values = list(filtered_labels), list(filtered_values)
    if sum_before_threshold - sum(filtered_values) > 0:
        filtered_labels.insert(len(filtered_labels), "Other")
        filtered_values.insert(
            len(filtered_values), sum_before_threshold - sum(filtered_values)
        )
    return filtered_labels, filtered_values


def get_label_discr(label: str) -> str:
    if label.split("_")[0] in ["read", "write"]:
        label = "_".join(label.split("_")[1:])
    if label.startswith("temp_"):
        label = label.split("_")[-1]
    return label


def add_labels_to_color_mapping(labels: list) -> None:
    for label in labels:
        label = get_label_discr(label)
        if label in COLOR_MAPPING:
            continue
        COLOR_MAPPING[label] = palette.hex_colors[
            min(len(COLOR_MAPPING), len(palette.hex_colors) - 1)
        ]


def generate_all_distribution_plots(
    output_directory: str, from_est: bool = True
) -> None:
    plots_fields = []
    if os.path.exists("distribution_plots.conf.csv"):
        with open("distribution_plots.conf.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) > 0 and row[0].startswith("#"):
                    continue
                plots_fields.append(row)
    with open(os.path.join(output_directory, "summary.json"), "r") as json_file:
        data = json.load(json_file)
        if from_est:
            classes_count = data["classes_estimated_all_jobs"]
        else:
            classes_count = data["classes_job_processed"]
    COLOR_MAPPING.clear()
    for p in plots_fields:
        if len(p) == 0:
            COLOR_MAPPING.clear()
            continue
        values = []
        plot_title = p.pop(0)
        labels = []
        for c in p:
            if c not in classes_count:
                print(f"Class {c} is not found in summary.json file")
                continue
            labels.append(c)
            values.append(classes_count[c])
        filtered_labels, filtered_values = filter_values(labels, values, 0.025)
        add_labels_to_color_mapping(filtered_labels)
        generate_donut_plot(
            plot_title.replace(" ", "\n"),
            os.path.join(
                output_directory, f'{plot_title.lower().replace(" ", "_")}.donut.svg'
            ),
            filtered_labels,
            filtered_values,
        )
        generate_stacked_plot(
            os.path.join(
                output_directory, f'{plot_title.lower().replace(" ", "_")}.bar.svg'
            ),
            filtered_labels,
            filtered_values,
        )


def generate_all_occurrences_plots(
    categorizer, traces_of_class: dict, estimate: bool
) -> None:
    plots_fields = []
    generate_co_occurrence(
        categorizer, traces_of_class, "all_classes", estimate=estimate
    )
    if os.path.exists("co_occurrences_maps.conf.csv"):
        with open("co_occurrences_maps.conf.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) == 0:
                    continue
                plots_fields.append(row)
    if len(plots_fields) % 2 == 1:
        raise RuntimeError(
            "co_occurrences_maps.conf.csv needs to have a pair number of lines as both x and y classes must be given"
        )
    i = 0
    while len(plots_fields):
        i += 1
        y_val = plots_fields.pop(0)
        x_val = plots_fields.pop(0)
        generate_co_occurrence(
            categorizer, traces_of_class, f"cust_{i}", y_val, x_val, estimate=estimate
        )


def generate_co_occurrence(
    categorizer,
    traces_of_class: dict,
    name: str,
    y_classes: list = None,
    x_classes: list = None,
    estimate: bool = True,
) -> None:
    all_classes = sorted(traces_of_class.keys())
    co_occurrence = defaultdict(lambda: defaultdict(float))
    class_sets = {cls: set(objs) for cls, objs in traces_of_class.items()}
    if not y_classes:
        y_classes = all_classes
    else:
        y_classes = [cl for cl in y_classes if cl in all_classes]
    if not x_classes:
        x_classes = all_classes
    else:
        x_classes = [cl for cl in x_classes if cl in all_classes]
    for class_y in y_classes:
        for class_x in x_classes:
            shared_objects = class_sets[class_y].intersection(class_sets[class_x])
            if estimate:
                co_occurrence[class_y][class_x] = sum(
                    [
                        len(categorizer.traces_of_hash[categorizer.get_exec_hash(t)])
                        for t in shared_objects
                    ]
                ) / sum(
                    [
                        len(categorizer.traces_of_hash[categorizer.get_exec_hash(t)])
                        for t in class_sets[class_y]
                    ]
                )
            else:
                co_occurrence[class_y][class_x] = len(shared_objects) / len(
                    class_sets[class_y]
                )
    df_matrix = pd.DataFrame(0.0, index=y_classes, columns=x_classes)
    for class_y in y_classes:
        for class_x in x_classes:
            df_matrix.loc[class_y, class_x] = co_occurrence[class_y][class_x]
    n_rows, n_cols = df_matrix.shape
    fig_width = n_cols * 0.75
    fig_height = n_rows * 0.75
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
    ax.imshow(df_matrix.values, cmap="Blues")
    ax.set_xticks(range(len(x_classes)))
    x_labels = [
        x.replace("read_", "r_").replace("write_", "w_").replace("metadata_", "mt_")
        for x in x_classes
    ]
    y_labels = [
        y.replace("read_", "r_").replace("write_", "w_").replace("metadata_", "mt_")
        for y in y_classes
    ]
    ax.set_xticklabels(x_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(y_classes)))
    ax.set_yticklabels(y_labels)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Co-occurring Class", labelpad=10)
    ax.set_ylabel("Base Class", labelpad=10)
    for i in range(len(y_classes)):
        for j in range(len(x_classes)):
            ax.text(
                j,
                i,
                f"{df_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if df_matrix.iloc[i, j] > 0.5 else "k",
            )
    max_xlabel_len = max([len(str(x)) for x in x_labels]) * 0.012
    max_ylabel_len = max([len(str(y)) for y in y_labels]) * 0.014
    print(x_labels)
    fig.set_size_inches(
        fig_width + max_ylabel_len + 0.1, fig_height + max_xlabel_len + 0.1
    )
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    plt.savefig(
        os.path.join(
            categorizer.output_directory, f"co_occurrences_heatmap_{name}.svg"
        ),
        bbox_inches="tight",
    )
    plt.close()
