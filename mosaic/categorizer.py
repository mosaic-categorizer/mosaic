import json
import os
import random
import signal
import sys
import time
from copy import copy

import dispy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from darshan.report import DarshanReport

from mosaic.trace_extractor import extract_traces

kill_switch, save_switch = False, False


class Categorizer:

    def __init__(self, trace_directory: str = 'none', output_directory: str = './out',
                 output_directory_graphs: str = 'graph', prune_inside_job: bool = False, mount: str = '/',
                 archives_directory: str = '', remove_unreadable_jobs: bool = True, prune_executions: bool = True,
                 dispy_nodes: str = 'localhost', remove_done: bool = True, load_from_pickle: bool = False):
        """
        @param trace_directory: directory where .darshan or .darshan.pkl.bz2 files are located (default: 'none')
        @param output_directory: directory where .json result files will be saved (default: ./out)
        @param output_directory_graphs: directory where activity graphs will be saved (default: ./{output_directory}/graphs)
        @param prune_inside_job: if multiple traces are from the same job, only keep one (default: False)
        @param mount: mounting point of PFS in darshan traces (default: /)
        @param archives_directory: directory where .tar files containing darshan files are located; if non-empty, archives will be extraced (default: '')
        @param remove_unreadable_jobs: remove unreadable traces (default: True)
        @param prune_executions: only keep one execution for each application (default: True)
        @param dispy_nodes: address of dispy nodes (default: 'localhost'); can be an array of str
        @param remove_done: remove traces already processed (default: True)
        @param load_from_pickle: load traces from .darshan.pkl.bz2 files instead of .darshan (default: False)
        """
        self.dispy_nodes = dispy_nodes.replace(' ', '').split(',')
        self.jobs = {}
        self.traces = []
        self.trace_directory = trace_directory
        self.trace_hash_cache = {}
        self.file_of_hash = {}
        self.trace_number = 0
        self.trace_stats = {}
        self.traces_to_process = []
        self.mount = mount
        self.output_directory = output_directory
        self.output_directory_graphs = os.path.join(output_directory, output_directory_graphs)
        self.recovered_results = []
        self.unique_applications = {}
        self.load_from_pickle = load_from_pickle

        if archives_directory != '':
            extract_traces(archives_directory, trace_directory, remove_unreadable_jobs)
        self.load_stats_json()
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if not os.path.exists(self.output_directory_graphs):
            os.makedirs(self.output_directory_graphs)

        if self.trace_directory == 'none':
            return

        self.enumerate_jobs()
        print(f'Found {len(self.jobs)} jobs')

        if load_from_pickle:
            self.trace_number = sum(map(lambda l: len(l), self.jobs.values()))
            self.traces = [trace for ll in self.jobs.values() for trace in ll]
            self.traces_to_process = copy(self.traces)
            return

        if prune_inside_job:
            print("Pruning executions in same job")
            self.prune_inside_jobs()
        self.trace_number = sum(map(lambda l: len(l), self.jobs.values()))
        print(f'Selected {self.trace_number} traces to analyze')
        if remove_unreadable_jobs:
            self.prune_unreadable_jobs()
        self.traces = [trace for ll in self.jobs.values() for trace in ll]
        if prune_executions:
            self.find_unique_applications()
        self.traces_to_process = copy(self.traces)
        if remove_done and 'processed' in self.trace_stats:
            self.remove_processed_traces()

    def prune_unreadable_jobs(self) -> None:
        """
        Removes all jobs that cannot be loaded
        """
        if 'removed_traces' in self.trace_stats.keys():
            print('Traces are already pruned')
        else:
            readable = self.find_readable_jobs()
            print(
                f'Can analyze {len(readable)} readable jobs '
                f'({"{:.2f}".format(len(readable) / len(self.jobs) * 100)})% of total jobs')
            jobs = set(self.jobs.keys())
            for job in jobs:
                if job not in readable:
                    del self.jobs[job]
            if len(readable) == len(self.jobs):
                self.trace_stats['removed_traces'] = []

    def find_unique_applications(self) -> None:
        """
        Keep only one trace per application in the dataset
        """
        if 'unique_applications' in self.trace_stats.keys():
            print('Unique applications are already found')
            self.unique_applications = self.trace_stats['unique_applications']
            self.traces = self.trace_stats['traces']
        else:
            self.prune_same_programs()
            print(f'Removed {self.trace_number - len(self.traces)} similar traces')
            self.trace_stats['unique_applications'] = self.unique_applications
            self.trace_stats['traces'] = self.traces
            with open(os.path.join(self.trace_directory, 'traces_stats.json'), 'w') as json_file:
                json.dump(self.trace_stats, json_file, indent=4)

    def remove_processed_traces(self) -> None:
        """
        Remove traces already categorized to prevent re-processing them
        """
        traces = list(self.traces)
        purged = 0
        for trace in traces:
            if self.get_trace_hash(trace) in self.trace_stats['processed']:
                self.traces_to_process.remove(trace)
                self.recovered_results.append((trace, self.recover_classifier_result(trace)))
                purged += 1
        print(f'Removed {purged} already processed traces')

    def load_stats_json(self) -> None:
        """
        Load traces_stats.json from previous execution if existing
        """
        trace_file = os.path.join(self.trace_directory, 'traces_stats.json')
        if os.path.isfile(trace_file):
            with open(trace_file, 'r') as json_file:
                self.trace_stats = json.load(json_file)

    def enumerate_jobs(self) -> None:
        """
        Find all job ids from dataset
        """
        if not os.path.isdir(self.trace_directory):
            raise RuntimeError(f'{self.trace_directory} is not a valid directory')
        files = os.listdir(self.trace_directory)
        if self.load_from_pickle:
            traces = [file for file in files if file.endswith('.darshan.pkl.bz2')]
        else:
            traces = [file for file in files if file.endswith('.darshan')]
        for trace in traces:
            job_id = trace.split('_')[-3]
            if job_id in self.jobs.keys():
                self.jobs[job_id].append(trace)
            else:
                self.jobs[job_id] = [trace]

    def prune_inside_jobs(self) -> None:
        """
        Only keep one execution for each job
        """
        for job in self.jobs.keys():
            if len(self.jobs[job]) == 1:
                continue
            else:
                distinct_names = set(map(lambda trace_name: '_'.join(trace_name.split('_')[1:][:-3]), self.jobs[job]))
                kept_files = []
                for distinct_name in distinct_names:
                    traces = list(
                        filter(lambda trace_name, name=distinct_name: '_'.join(trace_name.split('_')[1:][:-3]) == name,
                               self.jobs[job]))
                    trace_sizes = list(
                        map(lambda trace_name: os.stat(os.path.join(self.trace_directory, trace_name)).st_size, traces))
                    kept_files.append(traces[trace_sizes.index(max(trace_sizes))])
                self.jobs[job] = kept_files

    def prune_same_programs(self) -> None:
        """
        Find all applications in the dataset and keep only one execution per application to be processed
        """
        similar_traces = {}
        for trace in self.traces:
            try:
                prog_hash = self.get_trace_hash(trace)
                if prog_hash not in similar_traces:
                    similar_traces[prog_hash] = []
                similar_traces[prog_hash].append(trace)
            except UnicodeDecodeError:
                print(f'Error opening {trace} file')
        for unique_application in similar_traces.keys():
            self.unique_applications[unique_application] = len(similar_traces[unique_application])
        for traces in similar_traces.values():
            trace_sizes = list(
                map(lambda trace_name: os.stat(os.path.join(self.trace_directory, trace_name)).st_size, traces))
            trace_to_keep = traces[trace_sizes.index(max(trace_sizes))]
            for trace in traces:
                if trace != trace_to_keep:
                    self.traces.remove(trace)

    def find_readable_jobs(self) -> set:
        """
        Find all files that can be loaded by PyDarshan
        @return: a set of all the readable traces
        """
        readable = set()
        print('Finding readable jobs:')
        i = 1
        for job in self.jobs.keys():
            sys.stdout.write(f'\r   Processing {i}/{len(self.jobs.keys())} jobs')
            sys.stdout.flush()
            i += 1
            if self.is_readable(job):
                readable.add(job)
        sys.stdout.write('\r   Done\n')
        sys.stdout.flush()
        return readable

    def is_readable(self, job: str) -> bool:
        """
        Tell if all traces from a job are readable
        @param job: job to analyze
        @return: True if all traces from job are readable, False otherwise
        """
        for trace in self.jobs[job]:
            try:
                DarshanReport(os.path.join(self.trace_directory, trace), read_all=False)
            except RuntimeError as _:
                return False
        return True

    def categorize_trace(self, trace: str) -> None:
        """
        Categorize a trace
        @param trace: path of trace to categorize
        """
        start = time.time()
        categorize_dispy(trace, os.path.abspath(self.trace_directory), os.path.abspath(self.output_directory),
                         os.path.abspath(self.output_directory_graphs), self.mount, os.getcwd(),
                         self.load_from_pickle)
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
        print(f'Categorizing {len(self.traces)} traces:')
        jobs = []

        self.sort_traces(sort_strategy)

        cluster = dispy.JobCluster(categorize_dispy,
                                   nodes=self.dispy_nodes,
                                   reentrant=True,
                                   ping_interval=1)
        for job in self.traces_to_process:
            job = cluster.submit(job, os.path.abspath(self.trace_directory), os.path.abspath(self.output_directory),
                                 os.path.abspath(self.output_directory_graphs), self.mount, os.getcwd(),
                                 self.load_from_pickle)
            job.id = job
            jobs.append(job)
        n_jobs = len(jobs)
        kill_switch, save_switch = False, False
        signal.signal(signal.SIGINT, stop_signal_handler)
        signal.signal(signal.SIGTSTP, save_signal_handler)
        last_processed_count = -1
        while True:
            pending_jobs = sum(j.status < 8 for j in jobs)
            status = []
            for i in range(5, 12):
                status.append(f'({sum(j.status == i for j in jobs)})')
            print(
                f"\rCompleted {n_jobs - pending_jobs} out of {n_jobs} ({format_duration(time.time() - start)}) {' '.join(status)}",
                end='', flush=True)
            if save_switch:
                save_switch = False
                if self.load_from_pickle:
                    last_processed_count = self.generate_report_from_pickle_files(jobs, last_processed_count)
                else:
                    last_processed_count = self.generate_report_with_est(jobs, last_processed_count)
            if 0 < timeout < time.time() - start or not pending_jobs or kill_switch:
                break
            time.sleep(update_rate)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if 0 < timeout < time.time() - start or kill_switch:
            print("\nTimeout exceeded, cancelling the remaining jobs")
            if self.load_from_pickle:
                last_processed_count = self.generate_report_from_pickle_files(jobs, last_processed_count)
            else:
                last_processed_count = self.generate_report_with_est(jobs, last_processed_count)
            print('Result exported')
            jobs_to_cancel = [job for job in jobs if job.status != dispy.DispyJob.Finished]
            for job in jobs_to_cancel:
                if job.status != dispy.DispyJob.Finished:
                    cluster.cancel(job)
        else:
            print("\nAll jobs completed within the timeout")
        cluster.close()
        if self.load_from_pickle:
            self.generate_report_from_pickle_files(jobs, last_processed_count)
        else:
            self.generate_report_with_est(jobs, last_processed_count)
        print(f'\nDone. Total time: {time.time() - start}')

    def simplify_trace(self, job: str) -> None:
        """
        Turns a trace into a .darshan.pkl.bz2 file
        @param job: path of trace to simplify
        """
        start = time.time()
        export_simplified_traces(job, os.path.abspath(self.trace_directory), os.path.abspath(self.output_directory),
                                 self.mount, os.getcwd())
        print(f'\nDone. Total time: {time.time() - start}')

    def simplify_all_traces(self, timeout: int = -1, sort_strategy: str = 'random', update_rate: int = 1) -> None:
        """
        Simplify all selected traces
        @param timeout: maximum processing time in seconds; -1 if unlimited
        @param sort_strategy: ordering strategy to process traces
        @param update_rate: progress update rate in seconds
        """
        global kill_switch
        start = time.time()
        print(f'Categorizing {len(self.traces)} traces:')
        jobs = []

        self.sort_traces(sort_strategy)

        cluster = dispy.JobCluster(export_simplified_traces,
                                   nodes=self.dispy_nodes,
                                   reentrant=True,
                                   ping_interval=1)
        for job in self.traces_to_process:
            job = cluster.submit(job, os.path.abspath(self.trace_directory), os.path.abspath(self.output_directory),
                                 self.mount, os.getcwd())
            job.id = job
            jobs.append(job)
        n_jobs = len(jobs)
        kill_switch = False
        signal.signal(signal.SIGINT, stop_signal_handler)
        while True:
            pending_jobs = sum(j.status < 8 for j in jobs)
            status = []
            for i in range(5, 12):
                status.append(f'({sum(j.status == i for j in jobs)})')
            print(
                f"\rCompleted {n_jobs - pending_jobs} out of {n_jobs} ({format_duration(time.time() - start)}) {' '.join(status)}",
                end='', flush=True)
            if 0 < timeout < time.time() - start or not pending_jobs or kill_switch:
                break
            time.sleep(update_rate)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if 0 < timeout < time.time() - start or kill_switch:
            print("\nTimeout exceeded, cancelling the remaining jobs")
            jobs_to_cancel = [job for job in jobs if job.status != dispy.DispyJob.Finished]
            for job in jobs_to_cancel:
                if job.status != dispy.DispyJob.Finished:
                    cluster.cancel(job)
        else:
            print("\nAll jobs completed within the timeout")
        cluster.close()
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
                                            key=lambda l: self.unique_applications[self.get_trace_hash(l)],
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

    def generate_report_from_pickle_files(self, dispy_jobs: list, last_processed_count: int = -1) -> int:
        """
        Generate .json report file when categorization is from .darshan.pkl.bz2 files
        @param dispy_jobs: Dispy jobs
        @param last_processed_count: number of processed traces from the previous report
        @return: number of processed traces in the newly generated report
        """
        done_jobs = [job() for job in dispy_jobs if job.status == dispy.DispyJob.Finished]
        n_job = len(dispy_jobs)
        n_done = len(done_jobs)
        n_canceled = n_job - n_done
        print(f'Processed successfully {n_done} traces over {n_job}')
        results = [(job[0], job[1]) for job in done_jobs if job[1]] + self.recovered_results
        if len(results) == last_processed_count:
            print('Results were already exported')
            return last_processed_count
        class_count_processed, hashes_of_class = {}, {}
        processed_programs = set()
        failed = 0
        for res in results:
            trace, classes = res
            if trace == 'failed':
                failed += 1
                continue
            if self.load_from_pickle:
                prog_hash = trace
            else:
                prog_hash = self.get_trace_hash(trace)
            processed_programs.add(prog_hash)
            for class_name in classes:
                if class_name not in class_count_processed:
                    class_count_processed[class_name] = 0
                    hashes_of_class[class_name] = []
                class_count_processed[class_name] += 1
                hashes_of_class[class_name].append(prog_hash)

        print(len(processed_programs))
        self.trace_stats['processed'] = list(processed_programs)
        with open(os.path.join(self.trace_directory, 'traces_stats.json'), 'w') as json_file:
            json.dump(self.trace_stats, json_file, indent=4)
        classes = list(class_count_processed.keys())
        categorized_traces = len(self.unique_applications) - n_canceled - failed
        for class_ in classes:
            class_count_processed[f'{class_}_distribution'] = round(class_count_processed[class_] / categorized_traces,
                                                                    3)

        with open(os.path.join(self.output_directory, 'summary.json'), "w") as file:
            json.dump({
                'infos': {
                    'total_executions': self.trace_number,
                    'readable_executions': sum(self.unique_applications.values()),
                    'unique_applications': len(self.unique_applications),
                    'canceled_categorizations': n_canceled,
                    'failed_categorizations': failed,
                },
                'classes_job_processed': class_count_processed,
            }, file, indent=4)
        hashes_of_class = dict(sorted(hashes_of_class.items(), key=lambda x: x[0]))
        all_hashes = list(set(sum(hashes_of_class.values(), [])))
        class_matrix = pd.DataFrame(
            {cls: [1 if h in hash_lst else 0 for h in all_hashes]
             for cls, hash_lst in hashes_of_class.items()},
            index=all_hashes)
        self.generate_heatmaps(class_matrix, hashes_of_class, False)
        if self.load_from_pickle:
            traces_of_class = {
                k: [item for item in v]
                for k, v in hashes_of_class.items()
            }
        else:
            traces_of_class = {
                k: [self.file_of_hash[item] for item in v]
                for k, v in hashes_of_class.items()
            }
        with open(os.path.join(self.output_directory, 'traces_of_class.json'), "w") as file:
            json.dump(traces_of_class, file, indent=4)
        return self.trace_stats['processed']

    def generate_report_with_est(self, jobs: list, last_processed_count: int = -1) -> int:
        """
        Generate .json report file when categorization is from .darshan files and produce global estimations
        @param dispy_jobs: Dispy jobs
        @param last_processed_count: number of processed traces from the previous report
        @return: number of processed traces in the newly generated report
        """
        done_jobs = [job() for job in jobs if job.status == dispy.DispyJob.Finished]
        n_job = len(jobs)
        n_done = len(done_jobs)
        n_canceled = n_job - n_done
        print(f'Processed successfully {n_done} traces over {n_job}')
        results = [(job[0], job[1]) for job in done_jobs if job[1]] + self.recovered_results
        if len(results) == last_processed_count:
            print('Results were already exported')
            return last_processed_count
        class_count_processed, class_count_all, hashes_of_class = {}, {}, {}
        processed_programs = set()
        failed = 0
        for res in results:
            trace, classes = res
            if trace == 'failed':
                failed += 1
                continue
            prog_hash = self.get_trace_hash(trace)
            processed_programs.add(prog_hash)
            for class_name in classes:
                if class_name not in class_count_processed:
                    class_count_processed[class_name] = 0
                    class_count_all[class_name] = 0
                    hashes_of_class[class_name] = []
                class_count_processed[class_name] += 1
                class_count_all[class_name] += self.unique_applications[prog_hash]
                hashes_of_class[class_name].append(prog_hash)

        print(len(processed_programs))
        self.trace_stats['processed'] = list(processed_programs)
        with open(os.path.join(self.trace_directory, 'traces_stats.json'), 'w') as json_file:
            json.dump(self.trace_stats, json_file, indent=4)
        classes = list(class_count_processed.keys())
        categorized_traces = len(self.unique_applications) - n_canceled - failed
        estimated_categorized_all_traces = sum(map(lambda prog: self.unique_applications[prog], processed_programs))
        for class_ in classes:
            class_count_processed[f'{class_}_distribution'] = round(class_count_processed[class_] / categorized_traces,
                                                                    3)
            class_count_all[f'{class_}_distribution'] = round(
                class_count_all[class_] / estimated_categorized_all_traces, 3)

        class_count_processed = dict(sorted(class_count_processed.items(), key=lambda x: x[0]))
        class_count_all = dict(sorted(class_count_all.items(), key=lambda x: x[0]))

        with open(os.path.join(self.output_directory, 'summary.json'), "w") as file:
            json.dump({
                'infos': {
                    'total_executions': self.trace_number,
                    'readable_executions': sum(self.unique_applications.values()),
                    'unique_applications': len(self.unique_applications),
                    'canceled_categorizations': n_canceled,
                    'failed_categorizations': failed,
                    'inferred_executions': estimated_categorized_all_traces
                },
                'classes_job_processed': class_count_processed,
                'classes_estimated_all_jobs': class_count_all
            }, file, indent=4)
        hashes_of_class = dict(sorted(hashes_of_class.items(), key=lambda x: x[0]))
        all_hashes = list(set(sum(hashes_of_class.values(), [])))
        class_matrix = pd.DataFrame(
            {cls: [1 if h in hash_lst else 0 for h in all_hashes]
             for cls, hash_lst in hashes_of_class.items()},
            index=all_hashes)
        class_matrix_estimated_all = pd.DataFrame(
            {cls: [self.unique_applications[h] if h in hash_lst else 0 for h in all_hashes]
             for cls, hash_lst in hashes_of_class.items()},
            index=all_hashes)
        self.generate_heatmaps(class_matrix, hashes_of_class, False)
        self.generate_heatmaps(class_matrix_estimated_all, hashes_of_class, True, '_estimated_all')
        traces_of_class = {
            k: [self.file_of_hash[item] for item in v]
            for k, v in hashes_of_class.items()
        }
        with open(os.path.join(self.output_directory, 'traces_of_class.json'), "w") as file:
            json.dump(traces_of_class, file, indent=4)
        return self.trace_stats['processed']

    def generate_heatmaps(self, class_matrix: pd.DataFrame, hashes_of_class: dict, estimate_all_traces: bool,
                          suffix: str = '') -> None:
        """
        Generate class association heatmaps
        @param class_matrix: matrix of traces sharing classes
        @param hashes_of_class: dictionary of trace hashes
        @param estimate_all_traces: estimate the results for the whole dataset or not
        @param suffix: heatmap file suffix
        """
        jaccard_sim_df = self.compute_jaccard_sim(hashes_of_class, estimate_all_traces)
        plt_size = max(5, int(len(hashes_of_class) / 1.25))
        plt.figure(figsize=(plt_size, plt_size))
        sns.heatmap(jaccard_sim_df, annot=True, square=True, cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, f'jaccard_heatmap{suffix}.svg'))
        corr = class_matrix.corr()
        plt.figure(figsize=(1.25 * plt_size, 1.25 * plt_size))
        sns.heatmap(corr, annot=True, square=True, cmap='coolwarm', cbar_kws={"shrink": .82})
        plt.savefig(os.path.join(self.output_directory, f'correlation_heatmap{suffix}.svg'))

    def compute_jaccard_sim(self, hashes_of_class: dict, estimated: bool) -> pd.DataFrame:
        """
        Compute Jaccard Similarity Indexes
        @param hashes_of_class: dictionary of trace hashes
        @param estimated: estimate the results for the whole dataset or not
        @return: dataframe of Jaccard Similarity Indexes
        """
        sim = {}
        for cl1 in hashes_of_class:
            values = []
            h1 = hashes_of_class[cl1]
            if estimated:
                h1_trace_count = sum(map(lambda h: self.unique_applications[h], h1))
            else:
                h1_trace_count = len(h1)
            for cl2 in hashes_of_class:
                h2 = hashes_of_class[cl2]
                if estimated:
                    h2_trace_count = sum(map(lambda h: self.unique_applications[h], h2))
                else:
                    h2_trace_count = len(h2)
                intersection = [h for h in h1 if h in h2]
                if estimated:
                    intersection_trace_count = sum(map(lambda h: self.unique_applications[h], intersection))
                else:
                    intersection_trace_count = len(intersection)
                values.append(intersection_trace_count / (h1_trace_count + h2_trace_count - intersection_trace_count))
            sim[cl1] = values
        return pd.DataFrame(sim, index=list(hashes_of_class))

    def get_trace_hash(self, trace: str) -> str:
        """
        Get the hash of a trace
        @param trace: trace to hash
        @return: hash
        """
        if trace not in self.trace_hash_cache:
            self.trace_hash_cache[
                trace] = f"{trace.split('_')[0]}-{DarshanReport(os.path.join(self.trace_directory, trace), read_all=False).metadata['exe']}"
            self.file_of_hash[self.trace_hash_cache[trace]] = trace
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
    print("\nUser has requested to stop processing")
    kill_switch = True


def save_signal_handler(_signum, _frame):
    """
    Handler of SIGSTP signal to save progress during processing
    """
    global save_switch
    print("\nUser has requested to save progress")
    save_switch = True


def categorize_dispy(trace: str, trace_directory: str, output_directory: str, output_directory_graphs: str,
                     mount: str, path: str, load_from_pickle: bool, metadata_spike_threshold: int = 10) -> (
        str, list):
    """
    Processing function when categorizing traces with Dispy jobs
    @param trace: trace to process
    @param trace_directory: directory containing trace files
    @param output_directory: directory to save output json files
    @param output_directory_graphs: directory to save output graphs
    @param mount: mounting point of PFS in darshan trace
    @param path: working path of Mosaic
    @param load_from_pickle: load .darshan.pkl.bz2 files instead of .darshan
    @param metadata_spike_threshold: threshold from which Mosaic consider a metadata spike as impactful
    @return: trace name, list of assigned classes
    """
    try:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        import sys
        sys.path.append(path)
        import bz2
        import _pickle as cpickle
        from darshan import DarshanReport
        import json
        from mosaic.classifier import classify_trace
        from mosaic.periodicity_finder import load_trace
        from mosaic.periodicity_finder import compute_metadata_stats, find_periodic_patterns
        from mosaic.serializer import serialize_dict
        from mosaic.visualizer import visualize
    except Exception as e:
        return f'failed (setup): {e}', []
    try:
        if load_from_pickle:
            with bz2.BZ2File(os.path.join(trace_directory, trace), 'rb') as f:
                job = cpickle.load(f)
        else:
            job = load_trace(DarshanReport(os.path.join(trace_directory, trace), read_all=True), trace)
        metadata = compute_metadata_stats(job, mount, metadata_spike_threshold)
        write_segments, write_job = find_periodic_patterns(job, 'write', mount)
        read_segments, read_job = find_periodic_patterns(job, 'read', mount)
        result = {'infos': write_job['infos'], 'classes': None, 'metadata': metadata, 'read': read_segments,
                  'write': write_segments}
        classes = classify_trace(result, len(read_segments) > 0, len(write_segments) > 0)
        result['classes'] = classes
        if len(write_segments) > 0 or len(read_segments) > 0:
            visualize(write_job, write_segments, classes['write_classes'], read_job, read_segments,
                      classes['read_classes'], output_directory_graphs, mount)
        with open(os.path.join(output_directory, trace + '.json'), "w") as file:
            json.dump(serialize_dict(result), file, indent=4)
    except Exception as e:
        print(' Error extracting patterns of trace', trace, e)
        return f'failed: {e}', []
    return trace, [class_list for category in classes.values() for class_list in category]


def export_simplified_traces(trace: str, trace_directory: str, output_directory: str, mount: str, path: str) -> (
        str, list):
    """
    Processing function when simplifying traces with Dispy jobs
    @param trace: trace to process
    @param trace_directory: directory containing trace files
    @param output_directory: directory to save output .darshan.pkl.bz2 files
    @param mount: mounting point of PFS in darshan trace
    @param path: working path of Mosaic
    @return: trace name
    """
    try:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        import sys
        sys.path.append(path)
        from darshan import DarshanReport
        import bz2
        import _pickle as cpickle
        from mosaic.periodicity_finder import load_trace
    except Exception as e:
        return f'failed (setup): {e}', []
    try:
        job = load_trace(DarshanReport(os.path.join(trace_directory, trace), read_all=True), trace, mount)
        with bz2.BZ2File(os.path.join(output_directory, trace + '.pkl.bz2'), 'wb') as f:
            cpickle.dump(job, f)
    except Exception as e:
        print(' Error extracting patterns of trace', trace, e)
        return f'failed: {e}', []
    return trace
