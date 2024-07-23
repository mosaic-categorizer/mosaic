# Mosaic - Merging Operations and SegmentAtion for I/o Categorization

Mosaic is a categorizer that describes I/O operations contained in Darshan files. 
It detects periodic operations, access temporality and estimates load from metadata requests.
It exports results as json files, interactive html plots as well as svg heatmaps.

## Getting Started

Mosaic is written with Python3.10.

Installing dependencies:
```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Starting Dispy for parallel processing
```shell
source venv/bin/activate
dispynode.py -i localhost -c -1 #use all available cores but one
```

Categorize Darshan traces
```python
import mosaic
categorizer = mosaic.Categorizer(archives_directory='$Directory of archives containing Darshan files$', trace_directory='$Directory where traces will be extracted$',output_directory='$Result directory$',mount='$PFS mounting point$')
categorizer.categorize_all_traces(sort_strategy='heaviest')
```
