import sys
sys.path.append('./')
from pysign.benchmark.basic import BenchmarkQM9, BenchmarkMD17, BenchmarkMD17Dynamics, BenchmarkNBody

# benchmark = BenchmarkQM9()
# benchmark = BenchmarkMD17()
# benchmark = BenchmarkMD17Dynamics()
benchmark = BenchmarkNBody()
benchmark.launch()