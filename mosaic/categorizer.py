import contextlib
import gzip
import json
import os
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

from mosaic.classifier import classify_trace
from mosaic.periodicity_finder import compute_metadata_stats, find_periodic_patterns
from mosaic.process_pool import ProcessPool
from mosaic.visualizer import visualize

kill_switch, save_switch = False, False


class Categorizer:

    def __init__(self, trace_directory: str = 'none', output_directory: str = './out', generate_graphs: bool = True,
                 mount: str = '/', prune_executions: bool = True):
        """
        @param trace_directory: directory where .darshan or .darshan.pkl.bz2 files are located (default: 'none')
        @param output_directory: directory where .json result files will be saved (default: ./out)
        @param generate_graphs: generate html graphs to show activity (default: True)
        @param mount: mounting point of PFS in darshan traces (default: /)
        @param prune_executions: only keep one execution for each application (default: True)
        """
        self.traces: list = []
        self.traces_to_process: list = []
        self.trace_hash_cache = {}
        self.trace_directory = trace_directory
        self.output_directory = output_directory
        self.generate_graphs = generate_graphs
        self.mount = mount

        Path(output_directory).mkdir(parents=True, exist_ok=True)
        if generate_graphs:
            Path(os.path.join(output_directory, 'graphs')).mkdir(parents=True, exist_ok=True)

        for file in os.listdir(trace_directory):
            if file.endswith('.json') or file.endswith('.json.gz'):
                self.traces.append(file)

        print(f'Found {len(self.traces)} traces in {trace_directory}')

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
                for t in tqdm(self.traces, file=sys.stdout, unit='traces'):
                    h = self.get_exec_hash(t)
                    if not h in self.traces_of_hash:
                        self.traces_of_hash[h] = []
                    self.traces_of_hash[h].append(t)
                with open(os.path.join(trace_directory, 'trace_hashes.json'), 'w') as f:
                    json.dump(self.traces_of_hash, f)
            if not self.traces_to_process:
                self.traces_to_process = list(random.choice(self.traces_of_hash[hg]) for hg in self.traces_of_hash)
                with open(os.path.join(output_directory, 'processed_traces.json'), 'w') as f:
                    json.dump(self.traces_to_process, f)
        else:
            self.traces_to_process = copy(self.traces)

        print(
            f'Selected {len(self.traces_to_process)} ({"{:.2f}".format(100 * len(self.traces_to_process) / len(self.traces))}%) traces to process')

        old_count = len(self.traces_to_process)
        existing_results = set()
        for file in os.listdir(output_directory):
            if file.endswith('.class.json'):
                existing_results.add(os.path.join(trace_directory, file.replace('.class.json', '')))
        self.traces_to_process = [trace for trace in self.traces_to_process if trace not in existing_results]

        if len(self.traces_to_process) != old_count:
            print(
                f'{old_count - len(self.traces_to_process)} traces were already processed, continuing with {len(self.traces_to_process)} traces')

    def categorize_trace(self, trace: str) -> None:
        """
        Categorize a trace
        @param trace: path of trace to categorize
        """
        start = time.time()
        categorize_trace(trace, os.path.abspath(self.output_directory), self.generate_graphs, self.mount, os.getcwd())
        print(f'\nDone. Total time: {time.time() - start}')

    def categorize_all_traces(self, timeout: int = -1, sort_strategy: str = 'random', update_rate: int = 1) -> None:
        """
        Categorize all selected traces
        @param timeout: maximum processing time in seconds; -1 if unlimited
        @param sort_strategy: ordering strategy to process traces
        @param update_rate: progress update rate in seconds
        """
        global kill_switch, save_switch
        start = time.time()
        print(f'Categorizing {len(self.traces_to_process)} traces:')

        self.sort_traces(sort_strategy)

        process_pool = ProcessPool(os.cpu_count() - 1)
        for trace in self.traces_to_process:
            process_pool.submit(categorize_trace, os.path.join(self.trace_directory, trace), os.path.abspath(self.output_directory), self.generate_graphs, self.mount, os.getcwd())
        kill_switch, save_switch = False, False
        signal.signal(signal.SIGINT, stop_signal_handler)
        signal.signal(signal.SIGTSTP, save_signal_handler)
        last_count = 0
        last_export_count = 0
        start_time = time.time()
        with tqdm(total=len(self.traces_to_process), file=sys.stdout, unit='traces') as pbar:
            while process_pool.is_running():
                time.sleep(update_rate)
                count = process_pool.get_n_done()
                if count > last_count:
                    pbar.update(count - last_count)
                    last_count = count
                else:
                    pbar.refresh()
                if save_switch:
                    last_export_count = self.generate_report_with_est_from_dispy(process_pool.get_result(), last_export_count)
                    save_switch = False
                if 0 < timeout < (time.time() - start_time) or kill_switch:
                    process_pool.kill()
                    kill_switch = False
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)
        self.generate_report_with_est_from_dispy(process_pool.get_result(), last_export_count)
        print(f'\nDone. Total time: {time.time() - start}')

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

    def generate_report_with_est_from_files(self) -> None:
        results = []
        for p_trace in self.traces_to_process:
            if os.path.isfile(os.path.join(self.output_directory, p_trace + '.class.json')):
                with open(os.path.join(self.output_directory, p_trace + '.class.json')) as f:
                    j = json.load(f)
                    results.append((p_trace,
                                    [class_list for category in j['classes'].values() for class_list in category]))
        self.generate_report_with_est(results, len(self.traces_to_process), len(results))

    def generate_report_with_est_from_dispy(self, jobs: list, last_processed_count: int = -1) -> int:
        """
        Generate .json report file when categorization is from .darshan files and produce global estimations
        @param jobs: Dispy jobs
        @param last_processed_count: number of processed traces from the previous report
        @return: number of processed traces in the newly generated report
        """
        results = [(job[0], job[1]) for job in jobs]
        print(f'Got results for {len(results)} traces. Exporting figures')
        if len(results) == last_processed_count:
            print('Results were already exported')
            return last_processed_count
        self.generate_report_with_est(results, len(self.traces_to_process), len(results))
        print('Export done')
        return len(results)

    def generate_report_with_est(self, results: list, n_selected: int, n_found: int) -> None:
        """
        Generate .json report file when categorization is from .darshan files and produce global estimations
        @param results: results from processed traces
        @param n_selected: number of selected traces to be processed
        @param n_found: number of found traces
        @return: number of processed traces in the newly generated report
        """
        n_canceled = n_selected - n_found
        class_count_processed, class_count_all, traces_of_class = {}, {}, {}
        processed_traces = set()
        failed = 0
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.output_directory + '/error.txt')
        for res in results:
            trace, classes = res
            if trace.startswith('failed'):
                failed += 1
                with open(self.output_directory + '/error.txt', 'a') as file:
                    file.write(trace + '\n')
                continue
            processed_traces.add(trace)
            for class_name in classes:
                if class_name not in class_count_processed:
                    class_count_processed[class_name] = 0
                    if hasattr(self, 'traces_of_hash'):
                        class_count_all[class_name] = 0
                    traces_of_class[class_name] = []
                class_count_processed[class_name] += 1
                if hasattr(self, 'traces_of_hash'):
                    class_count_all[class_name] += len(self.traces_of_hash[self.get_exec_hash(trace)])
                traces_of_class[class_name].append(trace)

        classes = list(class_count_processed.keys())
        categorized_traces = len(self.traces_to_process) - n_canceled - failed
        if hasattr(self, 'traces_of_hash'):
            estimated_categorized_all_traces = sum(
                map(lambda prog: len(self.traces_of_hash[self.get_exec_hash(trace)]), processed_traces))
        for class_ in classes:
            class_count_processed[f'{class_}_distribution'] = round(class_count_processed[class_] / categorized_traces,
                                                                    3)
            if hasattr(self, 'traces_of_hash'):
                class_count_all[f'{class_}_distribution'] = round(
                    class_count_all[class_] / estimated_categorized_all_traces, 3)

        class_count_processed = dict(sorted(class_count_processed.items(), key=lambda x: x[0]))
        if hasattr(self, 'traces_of_hash'):
            class_count_all = dict(sorted(class_count_all.items(), key=lambda x: x[0]))

        with open(os.path.join(self.output_directory, 'summary.json'), "w") as file:
            summary = {
                'infos': {
                    'total_traces': len(self.traces),
                    'processed_traces': len(self.traces_to_process),
                    'canceled_categorizations': n_canceled,
                    'failed_categorizations': failed,

                },
                'classes_job_processed': class_count_processed
            }
            if hasattr(self, 'traces_of_hash'):
                summary['classes_estimated_all_jobs'] = class_count_all
                summary['infos']['inferred_executions'] = estimated_categorized_all_traces
            json.dump(summary, file, indent=4)
        traces_of_class = dict(sorted(traces_of_class.items(), key=lambda x: x[0]))
        all_traces = list(set(sum(traces_of_class.values(), [])))
        class_matrix = pd.DataFrame(
            {cls: [1 if t in trace_lst else 0 for t in all_traces]
             for cls, trace_lst in traces_of_class.items()},
            index=all_traces)
        self.generate_heatmaps(class_matrix, traces_of_class, False)
        if hasattr(self, 'traces_of_hash'):
            class_matrix_estimated_all = pd.DataFrame(
                {cls: [len(self.traces_of_hash[self.get_exec_hash(t)]) if t in trace_lst else 0 for t in all_traces]
                 for cls, trace_lst in traces_of_class.items()},
                index=all_traces)
            self.generate_heatmaps(class_matrix_estimated_all, traces_of_class, True, '_estimated_all')
        with open(os.path.join(self.output_directory, 'traces_of_class.json'), "w") as file:
            json.dump(traces_of_class, file, indent=4)

    def generate_heatmaps(self, class_matrix: pd.DataFrame, traces_of_class: dict, estimate_all_traces: bool,
                          suffix: str = '') -> None:
        """
        Generate class association heatmaps
        @param class_matrix: matrix of traces sharing classes
        @param traces_of_class: dictionary of traces having a class
        @param estimate_all_traces: estimate the results for the whole dataset or not
        @param suffix: heatmap file suffix
        """
        jaccard_sim_df = self.compute_jaccard_sim(traces_of_class, estimate_all_traces)
        plt_size = max(5, int(len(traces_of_class) / 1.25))
        plt.figure(figsize=(plt_size, plt_size))
        sns.heatmap(jaccard_sim_df, annot=True, square=True, cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, f'jaccard_heatmap{suffix}.svg'))
        corr = class_matrix.corr()
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
        for cl1 in traces_of_class:
            values = []
            t1 = traces_of_class[cl1]
            if estimated:
                t1_trace_count = sum(map(lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]), t1))
            else:
                t1_trace_count = len(t1)
            for cl2 in traces_of_class:
                t2 = traces_of_class[cl2]
                if estimated:
                    t2_trace_count = sum(map(lambda t: len(self.traces_of_hash[self.get_exec_hash(t)]), t2))
                else:
                    t2_trace_count = len(t2)
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
            if trace.endswith('.json'):
                with open(trace, 'r') as f:
                    job = json.load(f)
            elif trace.endswith('.json.gz'):
                with gzip.open(os.path.join(self.trace_directory, trace), 'r') as f:
                    job = json.load(f)
            self.trace_hash_cache[trace] = f'{job["metadata"]["uid"]}{job["metadata"]["exe"]}{len(job["traceEvents"])}'
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


def categorize_trace(trace: str, output_directory: str, output_graphs: bool, mount: str, path: str,
                     metadata_spike_threshold: int = 10) -> (str, list):
    """
    Processing function when categorizing traces with Dispy jobs
    @param trace: trace to process
    @param output_directory: directory to save output json files
    @param output_graphs: output html graphs of trace
    @param mount: mounting point of PFS in darshan trace
    @param path: working path of Mosaic
    @param metadata_spike_threshold: threshold from which Mosaic consider a metadata spike as impactful
    @return: trace name, list of assigned classes
    """
    try:
        if os.path.isfile(os.path.join(output_directory, trace.split('/')[-1] + '.class.json')):
            with open(os.path.join(output_directory, trace.split('/')[-1] + '.class.json'), "r") as file:
                classes = json.load(file)['classes']
            return trace, [class_list for category in classes.values() for class_list in category]
        if trace.endswith('.json'):
            with open(trace, 'r') as f:
                job = json.load(f)
        elif trace.endswith('.json.gz'):
            with gzip.open(trace, 'r') as f:
                job = json.load(f)
        else:
            raise NotImplementedError(f'Unsupported trace format: {trace}')
        metadata = compute_metadata_stats(job, mount, metadata_spike_threshold)
        write_segments, write_job = find_periodic_patterns(job, 'write', mount)
        read_segments, read_job = find_periodic_patterns(job, 'read', mount)
        result = {'infos': write_job['metadata'], 'classes': None, 'metadata': metadata, 'read': read_segments,
                  'write': write_segments}
        classes = classify_trace(result, len(read_segments) > 0, len(write_segments) > 0)
        result['classes'] = classes
        if output_graphs and (len(write_segments) > 0 or len(read_segments) > 0):
            visualize(write_job, write_segments, classes['write_classes'], read_job, read_segments,
                      classes['read_classes'], os.path.join(output_directory, 'graphs'), mount)
        with open(os.path.join(output_directory, trace.split('/')[-1] + '.class.json'), "w") as file:
            json.dump(result, file, indent=4)
    except Exception as e:
        print(' Error extracting patterns of trace ', trace, ' ', e, file=sys.stderr)
        return f'failed to process {trace}: {e}', []
    return trace, [class_list for category in classes.values() for class_list in category]