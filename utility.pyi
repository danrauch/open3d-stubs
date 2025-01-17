from typing import Iterable, Union, Optional, overload
from collections.abc import Iterator
from enum import Enum
from numpy import float64, int32
from numpy.typing import ArrayLike, NDArray


class DoubleVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[DoubleVector, NDArray[float64], Iterable]) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> float: ...
    def __setitem__(self, key, value: ArrayLike) -> DoubleVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[float]: ...
    def append(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: float) -> int: ...
    def extend(self, L: Union[DoubleVector, Iterable]) -> None: ...
    def insert(self, i: int, x: float) -> None: ...
    def pop(self, i: Optional[int] = None) -> float: ...
    def remove(self, x: float) -> None: ...


class IntVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[IntVector, NDArray[int32], Iterable]) -> None: ...
    def __getitem__(self, key) -> int: ...
    def __setitem__(self, key, value: ArrayLike) -> IntVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def append(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[IntVector, Iterable]) -> None: ...
    def insert(self, i: int, x: int) -> None: ...
    def pop(self, i: Optional[int] = None) -> int: ...
    def remove(self, x: int) -> None: ...


class Matrix4dVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Matrix4dVector, NDArray[float64], Iterable]) -> None: ...
    def __getitem__(self, key) -> NDArray[float64]: ...
    def __setitem__(self, key, value: ArrayLike) -> Matrix4dVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[float64]]: ...
    def append(self, x: NDArray[float64]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Matrix4dVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[float64]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[float64]: ...
    def remove(self, x: NDArray[float64]) -> None: ...


class Vector2dVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Vector2dVector, NDArray[float64], Iterable]) -> None: ...
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> NDArray[float64]: ...
    def __setitem__(self, key, value: ArrayLike) -> Vector2dVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[float64]]: ...
    def append(self, x: NDArray[float64]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Vector2dVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[float64]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[float64]: ...
    def remove(self, x: NDArray[float64]) -> None: ...


class Vector2iVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Vector2iVector, NDArray[int32], Iterable]) -> None: ...
    def __getitem__(self, key) -> NDArray[int32]: ...
    def __setitem__(self, key, value: ArrayLike) -> Vector2iVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[int32]]: ...
    def append(self, x: NDArray[int32]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Vector2iVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[int32]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[int32]: ...
    def remove(self, x: NDArray[int32]) -> None: ...


class Vector3dVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Vector3dVector, NDArray[float64], Iterable]) -> None: ...
    def __getitem__(self, key) -> NDArray[float64]: ...
    def __setitem__(self, key, value: ArrayLike) -> Vector3dVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[float64]]: ...
    def append(self, x: NDArray[float64]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Vector3dVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[float64]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[float64]: ...
    def remove(self, x: NDArray[float64]) -> None: ...


class Vector3iVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Vector3iVector, NDArray[int32], Iterable]) -> None: ...
    def __getitem__(self, key) -> NDArray[int32]: ...
    def __setitem__(self, key, value: ArrayLike) -> Vector3iVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[int32]]: ...
    def append(self, x: NDArray[int32]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Vector3iVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[int32]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[int32]: ...
    def remove(self, x: NDArray[int32]) -> None: ...


class Vector4iVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: Union[Vector4iVector, NDArray[int32], Iterable]) -> None: ...
    def __getitem__(self, key) -> NDArray[int32]: ...
    def __setitem__(self, key, value: ArrayLike) -> Vector4iVector: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[NDArray[int32]]: ...
    def append(self, x: NDArray[int32]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[Vector4iVector, Iterable]) -> None: ...
    def insert(self, i: int, x: NDArray[int32]) -> None: ...
    def pop(self, i: Optional[int] = None) -> NDArray[int32]: ...
    def remove(self, x: NDArray[int32]) -> None: ...


class VerbosityLevel(Enum):
    Debug = ...
    Error = ...
    Info = ...
    Warning = ...


class VerbosityContextManager:
    def __init__(self, level: VerbosityLevel) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args, **kwargs): ...


def get_verbosity_level() -> VerbosityLevel: ...
def set_verbosity_level(verbosity_level: VerbosityLevel) -> None: ...
def reset_print_function() -> None: ...
