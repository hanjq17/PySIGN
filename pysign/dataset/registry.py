from ..utils import RegistryBase
import typing as _typing
from torch_geometric.data.dataset import Dataset


class DatasetRegistry(RegistryBase):
    @classmethod
    def register_dataset(cls, dataset_name: str) -> _typing.Callable[
        [_typing.Type[Dataset]], _typing.Type[Dataset]
    ]:
        def register_dataset_cls(dataset: _typing.Type[Dataset]):
            if not issubclass(dataset, Dataset):
                raise TypeError
            else:
                cls[dataset_name] = dataset
                return dataset

        return register_dataset_cls

    @classmethod
    def get_dataset(cls, dataset_name: str) -> _typing.Type[Dataset]:
        if dataset_name not in cls:
            raise NotImplementedError('Unknown dataset', dataset_name)
        else:
            return cls[dataset_name]