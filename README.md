# Mosaic - Merging Operations and SegmentAtion for I/o Categorization

Mosaic is a categorizer that describes I/O operations contained in I/O trace files.
It is designed to process large trace datasets collected at machine-scale to uderstand the I/O workload behavior.
It processes TEF files, a json based format, and Mosaic provides a function to convert Darshan traces to TEF.
It detects periodic operations, access temporalities and estimates load from metadata requests.
It exports results as json files, interactive html plots as well as svg heatmaps.

## Getting Started

Mosaic for Python3.10 and later.

## Installing dependencies:

```shell
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Import Mosaic

```python
import mosaic
```

## Convert Darshan traces to TEF

Mosaic processes TEF files (json list of events) to make it agnostic to the tool used to collect traces.
It provides a script to automatically convert Darshan files to TEF. Both standard Darshan traces and DXT traces are supported.

```python
mosaic.generate_traces_from_directory('$DIR_WITH_DARSHAN_FILES', '$OUTPUT_DIR', '$PFS_MOUNT_POINT')
```

The third argument allows filtering the I/O operations to only keep those on the PFS.
For instance, for Polaris, the work PFSs are mounted on `/lus`. Setting the 3rd argument to `/lus` will only keep the I/O operations reading / writing data to the PFSs.

# Categorize TEF traces

```python
categorizer = mosaic.Categorizer(trace_directory='$DIR_TEF_FILES', output_directory='$OUT_DIR', mount='$PFS_MOUNT_POINT', prune_executions=False, duration_threshold=60)
categorizer.categorize_all_traces()
```

Mosaic can detect executions that are likely to have the same patterns (same job cmd, same user, similar amount of bytes in the I/O operations) to reduce the number of traces to process.
Setting it to `False` allows the exhaustive processing of input TEF files.

The `duration_threshold` argument allows processing only traces with longer runtime than the provided threshold. In this example, only traces representing jobs longer than 60s will be processed.
