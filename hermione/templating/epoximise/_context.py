import contextlib
import threading
from contextlib import contextmanager
from collections.abc import MutableMapping
import os
from typing import Any, Iterator


from ._util import DotDict


class Context(DotDict, threading.local):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        init = {}
        init.update(dict(*args, **kwargs))
        super().__init__(init)

    @contextlib.contextmanager
    def __call__(self, *args: MutableMapping, **kwargs: Any) -> Iterator["Context"]:
        previous_context = self.__dict__.copy()
        try:
            new_context = dict(*args, **kwargs)
            self.update(new_context)
            yield self
        finally:
            self.clear()
            self.update(previous_context)


context = Context()
