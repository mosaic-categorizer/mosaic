import sys
import mosaic

timeout = -1

if len(sys.argv) > 1:
    timeout = int(sys.argv[1])

analyzer = mosaic.Categorizer(trace_directory='/dataset/', output_directory ='/result/tef_res', mount ='/mnt', prune_executions=False, duration_threshold=60)
analyzer.categorize_all_traces(timeout = timeout, sort_strategy ='heaviest')
analyzer.generate_mongodb_export()
