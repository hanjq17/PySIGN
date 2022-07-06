import typing as _typing


class _RegistryUtility:
    @classmethod
    def to_unique_identifier(cls, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError
        import re
        return re.sub("[^A-Za-z0-9]", '', name.strip()).lower()


class _RegistryMetaclass(type):
    def __getitem__(cls, k: str):
        __identifier: str = _RegistryUtility.to_unique_identifier(k)
        if __identifier in cls.__registry:
            return cls.__registry[__identifier][0]
        else:
            raise KeyError(k)

    def __setitem__(cls, k: str, v: type) -> None:
        if not (isinstance(k, str) and isinstance(v, type)):
            raise TypeError
        __identifier: str = _RegistryUtility.to_unique_identifier(k)
        if __identifier in cls.__registry:
            if not (
                    v == cls.__registry[__identifier][0] and
                    id(v) == id(cls.__registry[__identifier][0])
            ):
                raise ValueError
            cls.__registry[__identifier][1].add(k)
        else:
            cls.__registry[__identifier] = (v, {k})

    def __delitem__(cls, k: str) -> None:
        if not isinstance(k, str):
            raise TypeError
        __identifier: str = _RegistryUtility.to_unique_identifier(k)
        if __identifier in cls.__registry:
            if k in cls.__registry[__identifier][1]:
                cls.__registry[__identifier][1].remove(k)
            if len(cls.__registry[__identifier][1]) == 0:
                del cls.__registry[__identifier]

    def __contains__(cls, item):
        if isinstance(item, str):
            __identifier: str = _RegistryUtility.to_unique_identifier(item)
            return __identifier in cls.__registry
        return False

    def __iter__(cls) -> _typing.Iterator[str]:
        results: _typing.MutableSequence[str] = []
        for __identifier, (_, names) in cls.__registry.items():
            results.extend(names)
        return iter(results)

    def __new__(
            mcs, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        return super(_RegistryMetaclass, mcs).__new__(
            mcs, name, bases, namespace
        )

    def __init__(
            cls, name: str, bases: _typing.Tuple[type, ...],
            namespace: _typing.Dict[str, _typing.Any]
    ):
        super(_RegistryMetaclass, cls).__init__(
            name, bases, namespace
        )
        cls.__registry: _typing.MutableMapping[
            str, _typing.Tuple[type, _typing.MutableSet[str]]
        ] = {}


class RegistryBase(metaclass=_RegistryMetaclass):
    @classmethod
    def to_unique_identifier(cls, name: str) -> str:
        return _RegistryUtility.to_unique_identifier(name)
