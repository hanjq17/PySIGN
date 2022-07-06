from ..utils import RegistryBase
import typing as _typing
from .basic import Benchmark


__all__ = ['BenchmarkRegistry']


class BenchmarkRegistry(RegistryBase):
    @classmethod
    def register_benchmark(cls, benchmark_name: str) -> _typing.Callable[
        [_typing.Type[Benchmark]], _typing.Type[Benchmark]
    ]:
        def register_benchmark_cls(benchmark: _typing.Type[Benchmark]):
            if not issubclass(benchmark, Benchmark):
                raise TypeError
            else:
                cls[benchmark_name] = benchmark
                return benchmark

        return register_benchmark_cls

    @classmethod
    def get_benchmark(cls, benchmark_name: str) -> _typing.Type[Benchmark]:
        if benchmark_name not in cls:
            raise NotImplementedError('Unknown benchmark', benchmark_name)
        else:
            return cls[benchmark_name]