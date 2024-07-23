import glob
import gzip
import json
import os
import shutil
import sys
import tarfile

import darshan


def extract_traces(path: str, dest: str, remove_unreadable: bool = True) -> None:
    """
    Extract all .darshan traces contained in tar.gz archives
    @param path: directory containing archive files
    @param dest: destination of extracted files
    @param remove_unreadable: remove all unreadable .darshan trace files
    """
    stats = {}
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is not a valid file or directory")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if os.path.isfile(path):
        if not path.endswith(".tar.gz"):
            raise RuntimeError(f"{path} is not a tar.gz archive")
        print('Extracting 1 archive')
        extract_archive(path, dest)
        uncompress_files(dest)
    if os.path.isdir(path):
        files = os.listdir(path)
        archives = [file for file in files if file.endswith(".tar.gz")]
        print(f'Extracting {len(archives)} archives:')
        i = 1
        for archive in archives:
            sys.stdout.write(f'\r   Processing archive {i}/{len(archives)}: {archive}')
            sys.stdout.flush()
            i += 1
            extract_archive(os.path.join(path, archive), dest)
        sys.stdout.write('\r   Extraction done\n')
        sys.stdout.flush()
        uncompress_files(dest)
    stats['total_traces'] = len(os.listdir(dest))
    print(f"Extracted {stats['total_traces']} traces")
    if remove_unreadable:
        total_traces, removed_traces = remove_unreadable_jobs(dest)
        print(
            f'Removed {removed_traces} unreadable traces over {total_traces} traces ({"{:.2f}".format(removed_traces / total_traces * 100)}%)')
        stats['removed_traces'] = removed_traces
    with open(os.path.join(dest, 'traces_stats.json'), 'w') as json_file:
        json.dump(stats, json_file, indent=4)


def remove_unreadable_jobs(directory: str) -> (int, int):
    """
    Remove all unreadable darshan files
    @param directory: directory containing files
    @return: total number of traces, number of unreadable traces
    """
    traces = glob.glob(f"{directory}/*.darshan")
    total_traces, removed_traces = len(traces), 0
    for trace in traces:
        try:
            darshan.DarshanReport(trace, read_all=False)
        except RuntimeError as _:
            removed_traces += 1
            os.remove(trace)
    return total_traces, removed_traces


def extract_archive(file: str, dest: str) -> None:
    """
    Extract one tar archive
    @param file: archive file
    @param dest: destination of extracted files
    """
    tar_file = tarfile.open(file)
    tar_file.extractall(dest)
    tar_file.close()


def uncompress_files(directory: str) -> None:
    """
    Uncompress .darshan.gz files contained in a directory
    @param directory: directory containing .gz files
    """
    files = os.listdir(directory)
    compressed_files = [file for file in files if file.endswith(".gz")]
    print(f'Decompressing {len(compressed_files)} files:')
    i = 0
    for compressed_file in compressed_files:
        sys.stdout.write(f'\r   Processing file {i}/{len(compressed_files)}: {compressed_file}')
        sys.stdout.flush()
        i += 1
        with gzip.open(os.path.join(directory, compressed_file), 'rb') as f_in:
            with open(os.path.join(directory, compressed_file[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(directory, compressed_file))
    sys.stdout.write('\r   Decompression done\n')
    sys.stdout.flush()
