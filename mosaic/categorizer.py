import contextlib
import gzip
import json
import os
import pathlib
import random
import signal
import sys
import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from mosaic.classifier import classify_trace, compute_file_temperatures_multiple, classify_multiple_temperatures
from mosaic.convert_tef_to_ftio import convert_to_ftio, get_main_frequencies
from mosaic.periodicity_finder import compute_metadata_stats, find_periodic_patterns
from mosaic.process_pool import ProcessPool
from mosaic.visualizer import visualize, generate_box_plots, generate_class_repartition_wrt_io, \
    generate_all_distribution_plots, generate_all_occurrences_plots

kill_switch = False


class Categorizer:

    def __init__(self, trace_directory: str = 'none', output_directory: str = './out', generate_graphs: bool = True,
                 mount: str = '/', prune_executions: bool = True, duration_threshold: int = -1, rank: int = -1, n_ranks: int = -1):
        """
        @param trace_directory: directory where .darshan or .darshan.pkl.bz2 files are located (default: 'none')
        @param output_directory: directory where .json result files will be saved (default: ./out)
        @param generate_graphs: generate html graphs to show activity (default: True)
        @param mount: mounting point of PFS in darshan traces (default: /)
        @param prune_executions: only keep one execution for each application (default: True)
        """
        self.traces_to_process: list = []
        self.trace_hash_cache = {}
        self.trace_directory = trace_directory
        self.output_directory = output_directory
        self.generate_graphs = generate_graphs
        self.mount = mount

        Path(output_directory).mkdir(parents=True, exist_ok=True)
        if generate_graphs:
            Path(os.path.join(output_directory, 'graphs')).mkdir(parents=True, exist_ok=True)

        traces = []
        for file in os.listdir(trace_directory):
            if file.endswith('.tef.json') or file.endswith('.json.gz'):
                traces.append(file)

        print(f'Found {len(traces)} traces in {trace_directory}')
        self.n_traces_in_dir = len(traces)

        compute_ban_traces_todo = duration_threshold > 0
        if os.path.isfile(os.path.join(trace_directory, 'banned_traces.json')):
            with open(os.path.join(trace_directory, 'banned_traces.json')) as f:
                banned_traces = json.load(f)
            if banned_traces['threshold'] == duration_threshold:
                compute_ban_traces_todo = False
                traces = list(set(traces) - set(banned_traces['banned']))
            else:
                pathlib.Path(os.path.join(trace_directory, 'banned_traces.json')).unlink(missing_ok=True)
                pathlib.Path(os.path.join(trace_directory, 'trace_hashes.json')).unlink(missing_ok=True)
                pathlib.Path(os.path.join(output_directory, 'processed_traces.json')).unlink(missing_ok=True)

        if compute_ban_traces_todo:
            pathlib.Path(os.path.join(trace_directory, 'trace_hashes.json')).unlink(missing_ok=True)
            pathlib.Path(os.path.join(output_directory, 'processed_traces.json')).unlink(missing_ok=True)
            banned_traces = self.ban_executions_too_short(traces, duration_threshold)
            traces = list(set(traces) - set(banned_traces))

        self.n_traces_considered = len(traces)
        if duration_threshold > 0:
            print(
                f'{self.n_traces_in_dir - self.n_traces_considered} traces are discarded because under {duration_threshold} seconds')

        if prune_executions:
            if os.path.isfile(os.path.join(trace_directory, 'trace_hashes.json')):
                print(f'Restoring traces hashes from {os.path.join(trace_directory, "trace_hashes.json")}')
                with open(os.path.join(trace_directory, 'trace_hashes.json')) as f:
                    self.traces_of_hash = json.load(f)
                    for h in self.traces_of_hash:
                        for t in self.traces_of_hash[h]:
                            self.trace_hash_cache[t] = h
                if os.path.isfile(os.path.join(output_directory, 'processed_traces.json')):
                    print(f'Restoring traces to process from {os.path.join(output_directory, "processed_traces.json")}')
                    with open(os.path.join(output_directory, "processed_traces.json")) as f:
                        self.traces_to_process = json.load(f)
            else:
                self.traces_of_hash = {}
                print('Generating hashes from traces')
                process_pool = ProcessPool(os.cpu_count() - 1)
                process_pool.batch_submit(traces, compute_trace_hash, self.trace_directory)
                process_pool.wait_completion()
                for res in process_pool.get_result():
                    trace, h = res
                    if not h in self.traces_of_hash:
                        self.traces_of_hash[h] = []
                    self.traces_of_hash[h].append(trace)
                with open(os.path.join(trace_directory, 'trace_hashes.json'), 'w') as f:
                    json.dump(self.traces_of_hash, f)
            if not self.traces_to_process:
                print('Select traces to process')
                self.traces_to_process = list(random.choice(self.traces_of_hash[hg]) for hg in self.traces_of_hash)
                with open(os.path.join(output_directory, 'processed_traces.json'), 'w') as f:
                    json.dump(self.traces_to_process, f)
        else:
            self.traces_to_process = copy(traces)
            with open(os.path.join(output_directory, 'processed_traces.json'), 'w') as f:
                json.dump(self.traces_to_process, f)

        if len(traces):
            print(
                f'Selected {len(self.traces_to_process)} ({"{:.2f}".format(100 * len(self.traces_to_process) / len(traces))}%) traces to process')

        old_count = len(self.traces_to_process)
        existing_results = set()
        for file in os.listdir(output_directory):
            if file.endswith('.class.json'):
                existing_results.add(file.replace('.class.json', ''))
        self.traces_to_process = [trace for trace in self.traces_to_process if trace not in existing_results]

        if len(self.traces_to_process) != old_count:
            print(
                f'{old_count - len(self.traces_to_process)} traces were already processed, continuing with {len(self.traces_to_process)} traces')

        if rank != -1 and n_ranks != -1:
            self.traces_to_process = sorted(self.traces_to_process)[rank::n_ranks]
            print(f'Processing {len(self.traces_to_process)} traces on rank {rank} out of {n_ranks}')

    def ban_executions_too_short(self, traces: list, threshold: int) -> list:
        print(f'Banning executions under {threshold} seconds')
        process_pool = ProcessPool(os.cpu_count() - 1)
        process_pool.batch_submit(traces, exec_is_too_short, self.trace_directory, threshold)
        process_pool.wait_completion()
        traces_too_short = []
        for res in process_pool.get_result():
            trace, too_short = res
            if too_short:
                traces_too_short.append(trace)
        with open(os.path.join(self.trace_directory, 'banned_traces.json'), 'w') as f:
            json.dump({'threshold': threshold, 'banned': traces_too_short}, f)
        return traces_too_short

    def clean_traces(self) -> None:
        if not os.path.isfile(os.path.join(self.output_directory, 'processed_traces.json')):
            raise Exception('No traces to process')
        with open(os.path.join(self.output_directory, "processed_traces.json")) as f:
            traces_to_process = json.load(f)
        print(f'{len(traces_to_process)} traces to process')
        existing_results = set()
        for file in os.listdir(self.output_directory):
            if file.endswith('.class.json'):
                existing_results.add(file.replace('.class.json', ''))
        print(f'Found {len(existing_results)} existing results')
        to_delete = set()
        for result in existing_results:
            if result not in traces_to_process:
                to_delete.add(result)
        print(f'Deleting {len(to_delete)} results not in the list of traces to process')

    def generate_mongodb_export(self):
        if not os.path.isfile(os.path.join(self.output_directory, 'processed_traces.json')):
            raise Exception('No traces to process')
        with open(os.path.join(self.output_directory, "processed_traces.json")) as f:
            traces_to_process = json.load(f)
        exported = []
        error, not_found = 0, 0
        estimate = hasattr(self, 'traces_of_hash')
        for trace in traces_to_process:
            if not os.path.isfile(os.path.join(self.output_directory, f'{trace}.class.json')):
                not_found += 1
                continue
            with open(os.path.join(self.output_directory, f'{trace}.class.json')) as f:
                j = json.load(f)
            if j['status'] == 'error':
                error += 1
                continue
            del j['file_temperatures']
            j['_id'] = self.get_exec_hash(trace)
            if estimate:
                j['representation_count'] = len(self.traces_of_hash[self.get_exec_hash(trace)])
            exported.append(j)
        with open(os.path.join(self.output_directory, 'mongo_export.json'), 'w') as f:
            json.dump(exported, f)
        print(f'Exported {len(exported)} results to mongo_export.json ({error} errors, {not_found} not found)')

    def categorize_trace(self, trace: str) -> None:
        """
        Categorize a trace
        @param trace: path of trace to categorize
        """
        start = time.time()
        categorize_trace(trace, os.path.abspath(self.output_directory), self.generate_graphs, self.mount)
        print(f'\nDone. Total time: {time.time() - start}')

    def categorize_all_traces(self, timeout: int = -1, sort_strategy: str = 'random', update_rate: int = 1,
                              n_proc=-1) -> None:
        """
        Categorize all selected traces
        @param timeout: maximum processing time in seconds; -1 if unlimited
        @param sort_strategy: ordering strategy to process traces
        @param update_rate: progress update rate in seconds
        """
        global kill_switch
        start = time.time()
        print(f'Categorizing {len(self.traces_to_process)} traces:')

        self.sort_traces(sort_strategy)

        process_pool = ProcessPool(os.cpu_count() - 1) if n_proc == -1 else ProcessPool(n_proc)
        process_pool.batch_submit([os.path.join(self.trace_directory, trace) for trace in self.traces_to_process],
                                  categorize_trace, os.path.abspath(self.output_directory), self.generate_graphs,
                                  self.mount)
        kill_switch = False
        signal.signal(signal.SIGINT, stop_signal_handler)
        last_count = 0
        start_time = time.time()
        with tqdm(total=len(self.traces_to_process), file=sys.stdout, unit='traces') as pbar:
            while process_pool.is_running():
                time.sleep(update_rate)
                count = process_pool.get_n_done()
                process_pool.submit_more_tasks(count)
                if count > last_count:
                    pbar.update(count - last_count)
                    last_count = count
                else:
                    pbar.refresh()
                if 0 < timeout < (time.time() - start_time) or kill_switch:
                    process_pool.kill()
                    kill_switch = False
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print(f'Finished processing the traces in {time.time() - start}s')

    def sort_traces(self, strategy: str) -> None:
        """
        Sort traces according to strategy
        @param strategy: strategy to apply
        """
        if strategy == 'random':
            random.shuffle(self.traces_to_process)
            return
        if strategy == 'most_significant':
            self.traces_to_process = sorted(self.traces_to_process,
                                            key=lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]),
                                            reverse=True)
            return
        trace_sizes = {}
        for trace in self.traces_to_process:
            trace_sizes[trace] = os.stat(os.path.join(self.trace_directory, trace)).st_size
        if strategy == 'heaviest':
            self.traces_to_process = sorted(self.traces_to_process, key=trace_sizes.get, reverse=True)
            return
        if strategy == 'lightest':
            self.traces_to_process = sorted(self.traces_to_process, key=trace_sizes.get, reverse=False)
            return
        raise NotImplementedError(f'{strategy} sort strategy not implemented')

    def generate_report_with_est_from_files(self, delete_on_error: bool = False) -> None:
        results = []
        with open(os.path.join(self.output_directory, "processed_traces.json")) as f:
            traces_to_process = json.load(f)
        t = time.time()
        mean_temp = {}
        hl = [1, 10, 60, 1800, 3600]
        for t_ in hl:
            mean_temp[t_] = []
        for p_trace in traces_to_process:
            if os.path.isfile(os.path.join(self.output_directory, p_trace + '.class.json')):
                with open(os.path.join(self.output_directory, p_trace + '.class.json')) as f:
                    j = json.load(f)
                    if j['status'] == 'error':
                        print(f'{p_trace} classification failed: {j["message"]}')
                        if delete_on_error:
                            os.remove(os.path.join(self.output_directory, p_trace + '.class.json'))
                            print(f'{p_trace} removed')
                        continue
                    total_io = sum([i['data_operated_avg'] * i['segments_cnt'] for i in (j['read'] + j['write'])])
                    results.append([p_trace,
                                    [class_list for category in j['classes'].values() for class_list in category],
                                    total_io])
                    for t_ in hl:
                        t_tot = 0
                        for temp in j['file_temperatures'][f'{t_}'].values():
                            t_tot += temp
                        if hasattr(self, 'traces_of_hash'):
                            mean_temp[t_].extend(
                                [t_tot for _ in range(len(self.traces_of_hash[self.get_exec_hash(p_trace)]))])
                        else:
                            mean_temp[t_].append(t_tot)
        with open(os.path.join(self.output_directory, 'heat_distr.json'), 'w') as f:
            f.write(json.dumps(mean_temp))
        for t_ in hl:
            print(f'Temp {t_}s - mean {sum(mean_temp[t_]) / len(mean_temp[t_]):.2f} max {max(mean_temp[t_]):.2f}')
        self.generate_report_with_est(results, len(traces_to_process), len(results))
        print(f'Result exported in {time.time() - t} seconds')

    def generate_report_with_est_from_dispy(self, jobs: list, last_processed_count: int = -1) -> int:
        """
        Generate .json report file when categorization is from .darshan files and produce global estimations
        @param jobs: Dispy jobs
        @param last_processed_count: number of processed traces from the previous report
        @return: number of processed traces in the newly generated report
        """
        print(f'Got results for {len(jobs)} traces. Exporting figures')
        if len(jobs) == last_processed_count:
            print('Results were already exported')
            return last_processed_count
        self.generate_report_with_est(jobs, len(self.traces_to_process), len(jobs))
        print('Export done')
        return len(jobs)

    def generate_report_with_est(self, results: list, n_selected: int, n_found: int) -> None:
        """
        Generate .json report file when categorization is from .darshan files and produce global estimations
        @param results: results from processed traces
        @param n_selected: number of selected traces to be processed
        @param n_found: number of found traces
        @return: number of processed traces in the newly generated report
        """
        n_canceled = n_selected - n_found
        class_count_processed, class_count_all, traces_of_class, io_sizes_per_class = {}, {}, {}, {}
        processed_traces = set()
        failed = 0
        estimate = hasattr(self, 'traces_of_hash')
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.output_directory + '/error.txt')
        for res in results:
            trace, classes, total_io = res
            if trace.startswith('failed'):
                failed += 1
                with open(self.output_directory + '/error.txt', 'a') as file:
                    file.write(trace + '\n')
                continue
            processed_traces.add(trace)
            if estimate:
                total_t_count = len(self.traces_of_hash[self.get_exec_hash(trace)])
            for class_name in classes:
                if class_name not in class_count_processed:
                    class_count_processed[class_name] = 0
                    io_sizes_per_class[class_name] = []
                    if estimate:
                        class_count_all[class_name] = 0
                    traces_of_class[class_name] = []
                class_count_processed[class_name] += 1
                if estimate:
                    for _ in range(total_t_count):
                        io_sizes_per_class[class_name].append(total_io)
                else:
                    io_sizes_per_class[class_name].append(total_io)
                if estimate:
                    class_count_all[class_name] += total_t_count
                traces_of_class[class_name].append(trace)
        classes = list(class_count_processed.keys())
        categorized_traces = n_selected - n_canceled - failed
        if estimate:
            estimated_categorized_all_traces = sum(
                map(lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]), processed_traces))
        for class_ in classes:
            class_count_processed[f'{class_}_distribution'] = round(class_count_processed[class_] / categorized_traces,
                                                                    3)
            if estimate:
                class_count_all[f'{class_}_distribution'] = round(
                    class_count_all[class_] / estimated_categorized_all_traces, 3)

        class_count_processed = dict(sorted(class_count_processed.items(), key=lambda x: x[0]))
        if estimate:
            class_count_all = dict(sorted(class_count_all.items(), key=lambda x: x[0]))

        with open(os.path.join(self.output_directory, 'summary.json'), "w") as file:
            summary = {
                'infos': {
                    'total_traces': self.n_traces_in_dir,
                    'processed_traces': n_selected,
                    'canceled_categorizations': n_canceled,
                    'failed_categorizations': failed,

                },
                'classes_job_processed': class_count_processed
            }
            if hasattr(self, 'traces_of_hash'):
                summary['classes_estimated_all_jobs'] = class_count_all
                summary['infos']['inferred_executions'] = estimated_categorized_all_traces
                if self.n_traces_in_dir != self.n_traces_considered:
                    summary['infos']['considered_executions'] = self.n_traces_considered
            json.dump(summary, file, indent=4)
        generate_box_plots(io_sizes_per_class, self.output_directory)
        generate_class_repartition_wrt_io(io_sizes_per_class, self.output_directory)
        generate_all_distribution_plots(self.output_directory, estimate)
        traces_of_class = dict(sorted(traces_of_class.items(), key=lambda x: x[0]))
        all_traces = list(set(sum(traces_of_class.values(), [])))
        generate_all_occurrences_plots(self, traces_of_class, estimate)
        self.generate_heatmaps(traces_of_class, False, False, all_traces)
        if estimate:
            self.generate_heatmaps(traces_of_class, True, False, all_traces, '_estimated_all')
        with open(os.path.join(self.output_directory, 'traces_of_class.json'), "w") as file:
            json.dump(traces_of_class, file, indent=4)

    def generate_heatmaps(self, traces_of_class: dict, estimate_all_traces: bool,
                          gen_correlation: bool, all_traces: list, suffix: str = '') -> None:
        """
        Generate class association heatmaps
        @param traces_of_class: dictionary of traces having a class
        @param estimate_all_traces: estimate the results for the whole dataset or not
        @param gen_correlation: tell to generate correlation heatmaps or not
        @param all_traces: list of all categorized traces
        @param suffix: heatmap file suffix
        """
        plt_size = max(5, int(len(traces_of_class) / 1.25))
        jaccard_sim_df = self.compute_jaccard_sim(traces_of_class, estimate_all_traces)
        plt.figure(figsize=(plt_size, plt_size))
        sns.heatmap(jaccard_sim_df, annot=True, square=True, cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, f'jaccard_heatmap{suffix}.svg'))
        if not gen_correlation:
            return
        if estimate_all_traces:
            corr = pd.DataFrame(
                {cls: [len(self.traces_of_hash[self.get_exec_hash(t)]) if t in trace_lst else 0 for t in all_traces]
                 for cls, trace_lst in traces_of_class.items()},
                index=all_traces).corr()
        else:
            corr = pd.DataFrame(
                {cls: [1 if t in trace_lst else 0 for t in all_traces]
                 for cls, trace_lst in traces_of_class.items()},
                index=all_traces).corr()
        plt.figure(figsize=(1.25 * plt_size, 1.25 * plt_size))
        sns.heatmap(corr, annot=True, square=True, cmap='coolwarm', cbar_kws={"shrink": .82})
        plt.savefig(os.path.join(self.output_directory, f'correlation_heatmap{suffix}.svg'))

    def compute_jaccard_sim(self, traces_of_class: dict, estimated: bool) -> pd.DataFrame:
        """
        Compute Jaccard Similarity Indexes
        @param traces_of_class: dictionary of traces having a class
        @param estimated: estimate the results for the whole dataset or not
        @return: dataframe of Jaccard Similarity Indexes
        """
        sim = {}
        len_of_cl = {}
        for cl in traces_of_class:
            if estimated:
                len_of_cl[cl] = sum(map(lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]), traces_of_class[cl]))
            else:
                len_of_cl[cl] = len(traces_of_class[cl])
        for cl1 in traces_of_class:
            values = []
            t1 = traces_of_class[cl1]
            t1_trace_count = len_of_cl[cl1]
            for cl2 in traces_of_class:
                t2 = traces_of_class[cl2]
                t2_trace_count = len_of_cl[cl2]
                intersection = [t for t in t1 if t in t2]
                if estimated:
                    intersection_trace_count = sum(
                        map(lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]), intersection))
                else:
                    intersection_trace_count = len(intersection)
                values.append(intersection_trace_count / (t1_trace_count + t2_trace_count - intersection_trace_count))
            sim[cl1] = values
        return pd.DataFrame(sim, index=list(traces_of_class))

    def get_exec_hash(self, trace: str) -> str:
        """
        Get the hash of a trace
        @param trace: trace to hash
        @return: hash
        """
        if trace not in self.trace_hash_cache:
            _, self.trace_hash_cache[trace] = compute_trace_hash(trace, self.trace_directory)
        return self.trace_hash_cache[trace]

    def recover_classifier_result(self, trace: str) -> list:
        """
        Load results from previous categorization
        @param trace: trace to reload
        @return: list of assigned classes
        """
        trace_file_path = os.path.join(self.output_directory, trace + '.json')
        if not os.path.isfile(trace_file_path):
            raise FileNotFoundError(f'Trace file {trace_file_path} does not exist')
        with open(trace_file_path, 'r') as json_file:
            result = json.load(json_file)
        return [class_list for category in result['classes'].values() for class_list in category]


def format_duration(seconds: float) -> str:
    """
    Format durations for progress output
    @param seconds: duration in seconds
    @return: string of formatted duration
    """
    days = int(seconds // (24 * 3600))
    hours = int((seconds % (24 * 3600)) // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    formatted_duration = ""
    if days:
        formatted_duration += f"{days}d "
    if hours:
        formatted_duration += f"{hours}h "
    if minutes:
        formatted_duration += f"{minutes}min "
    if seconds:
        formatted_duration += f"{seconds}s"

    return formatted_duration.rstrip(', ')


def stop_signal_handler(_signum, _frame):
    """
    Handler of SIGINT signal to save progress before quitting
    """
    global kill_switch
    print('\nUser has requested to stop processing')
    kill_switch = True


def save_signal_handler(_signum, _frame):
    """
    Handler of SIGSTP signal to save progress during processing
    """
    global save_switch
    print('\nUser has requested to save progress')
    save_switch = True


def compute_trace_hash(trace: str, trace_directory: str) -> (str, str):
    if trace.endswith('.json'):
        with open(os.path.join(trace_directory, trace), 'r') as f:
            job = json.load(f)
    elif trace.endswith('.json.gz'):
        with gzip.open(os.path.join(trace_directory, trace), 'r') as f:
            job = json.load(f)
    n_bytes = sum([x["args"]["count"] for x in job["traceEvents"] if x["ph"] == "X"])
    n_bytes = n_bytes - (n_bytes % 1024)
    return trace, f'{job["metadata"]["uid"]}_{job["metadata"]["exe"]}_{len(job["traceEvents"])}_{n_bytes}'


def exec_is_too_short(trace: str, trace_directory: str, threshold: int) -> (str, bool):
    try:
        if trace.endswith('.json'):
            with open(os.path.join(trace_directory, trace), 'r') as f:
                job = json.load(f)
        elif trace.endswith('.json.gz'):
            with gzip.open(os.path.join(trace_directory, trace), 'r') as f:
                job = json.load(f)
    except Exception as e:
        print(f'Error while loading trace {trace}: {e}')
        raise Exception(f'Error while loading trace {trace}: {e}')
    return trace, job['metadata']['run_time'] < threshold


def categorize_trace(trace: str, output_directory: str, output_graphs: bool, mount: str,
                     metadata_spike_threshold: int = 10) -> list:
    """
    Processing function when categorizing traces with Dispy jobs
    @param trace: trace to process
    @param output_directory: directory to save output json files
    @param output_graphs: output html graphs of trace
    @param mount: mounting point of PFS in darshan trace
    @param metadata_spike_threshold: threshold from which Mosaic consider a metadata spike as impactful
    @return: trace name, list of assigned classes
    """
    try:
        if os.path.isfile(os.path.join(output_directory, trace.split('/')[-1] + '.class.json')):
            with open(os.path.join(output_directory, trace.split('/')[-1] + '.class.json'), "r") as file:
                j = json.load(file)
                classes = j['classes']
                total_io = sum([i['metadata_operations_avg'] * i['segments_cnt'] for i in (j['read'] + j['write'])])
            return [trace, [class_list for category in classes.values() for class_list in category], total_io]
        if trace.endswith('.json'):
            with open(trace, 'r') as f:
                job = json.load(f)
        elif trace.endswith('.json.gz'):
            with gzip.open(trace, 'r') as f:
                job = json.load(f)
        else:
            raise NotImplementedError(f'Unsupported trace format: {trace}')
        metadata = compute_metadata_stats(job, mount, metadata_spike_threshold)
        file_temperatures = compute_file_temperatures_multiple(job['traceEvents'], [1, 10, 60, 1800, 3600])
        ftio_data = convert_to_ftio(job)
        ftio_res = {}
        ftio_duration = 0
        if 'write' in ftio_data:
            start_ts = time.time()
            period = get_main_frequencies(ftio_data['write'], trace.split('/')[-1])
            ftio_duration += time.time() - start_ts
            if period > 0:
                ftio_res['write'] = period
        if 'read' in ftio_data:
            start_ts = time.time()
            period = get_main_frequencies(ftio_data['read'], trace.split('/')[-1])
            ftio_duration += time.time() - start_ts
            if period > 0:
                ftio_res['read'] = period
        start_ts = time.time()
        clustering_write_segments = find_periodic_patterns(job, 'write',
                                                           mount) if 'write' in ftio_data and 'write' not in ftio_res else []
        clustering_read_segments = find_periodic_patterns(job, 'read',
                                                          mount) if 'read' in ftio_data and 'read' not in ftio_res else []
        clustering_duration = time.time() - start_ts
        result = {'status': 'success', 'infos': job['metadata'], 'classes': None, 'metadata': metadata,
                  'read': clustering_read_segments, 'write': clustering_write_segments, 'ftio': ftio_res,
                  'file_temperatures': file_temperatures,
                  'debug_durations': {'clustering': clustering_duration, 'ftio': ftio_duration}}
        classes = classify_trace(job, result)
        classes['temperatures'] = classify_multiple_temperatures(file_temperatures)
        result['classes'] = classes
        if output_graphs and (len(clustering_write_segments) > 0 or len(clustering_read_segments) > 0):
            pass
            visualize(job, clustering_write_segments, classes['write_classes'], clustering_read_segments,
                      classes['read_classes'], os.path.join(output_directory, 'graphs'), mount)
        with open(os.path.join(output_directory, trace.split('/')[-1] + '.class.json'), "w") as file:
            json.dump(result, file, indent=4)
    except Exception as e:
        sys.stderr.write(f'Error processing {trace}: {str(e)}\n')
        result = {'status': 'error', 'message': str(e)}
        with open(os.path.join(output_directory, trace.split('/')[-1] + '.class.json'), "w") as file:
            json.dump(result, file, indent=4)
        return [f'failed to process {trace}: {e}', [], -1]
    return [trace, [class_list for category in classes.values() for class_list in category]]
