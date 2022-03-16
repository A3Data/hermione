import collections
from collections.abc import MutableMapping
from typing import Any, Generator, Iterable, Iterator, Union, cast, TypeVar, Type


DictLike = Union[dict, "DotDict"]
D = TypeVar("D", bound=Union[dict, MutableMapping])


class DotDict(MutableMapping):
    def __init__(self, init_dict: DictLike = None, **kwargs: Any):
        # a DotDict could have a key that shadows `update`
        if init_dict:
            super().update(init_dict)
        super().update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]  # __dict__ expects string keys

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __setattr__(self, attr: str, value: Any) -> None:
        self[attr] = value

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__.keys())

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __repr__(self) -> str:
        if len(self) > 0:
            return "<{}: {}>".format(
                type(self).__name__, ", ".join(sorted(repr(k) for k in self.keys()))
            )
        else:
            return "<{}>".format(type(self).__name__)

    def copy(self) -> "DotDict":
        return type(self)(self.__dict__.copy())
