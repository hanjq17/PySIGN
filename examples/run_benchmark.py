import argparse
import sys
sys.path.append('./')
from pysign.benchmark import BenchmarkRegistry


parser = argparse.ArgumentParser(description='BenchmarkDemo')
parser.add_argument('--benchmark', '-b', type=str,
                    help='benchmark name, currently supporting:' + ','.join([name for name in BenchmarkRegistry.__iter__()]))
args = parser.parse_args()

benchmark_fn = BenchmarkRegistry.get_benchmark(args.benchmark)
benchmark = benchmark_fn()
benchmark.launch()
