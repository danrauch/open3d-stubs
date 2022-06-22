from __future__ import annotations

from enum import Enum
from typing import (
    Iterable,
    Optional,
    Sequence,
    overload,
    Union,
    ClassVar,
    Tuple,
    List
)
from numpy import ArrayLike, ndarray


class Blob:
    def __init__(self, *args, **kwargs) -> None: ...


class Device:
    class DeviceType(Enum):
        CPU = ...
        CUDA = ...

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, device_type: Union[str, DeviceType], device_id: int) -> None: ...
    @overload
    def __init__(self, type_colon_id: str) -> None: ...
    def get_id(self) -> int: ...
    def get_type(self) -> DeviceType: ...


class DtypeCode(Enum):
    Bool = ...
    Float = ...
    Int = ...
    Object = ...
    UInt = ...
    Undefined = ...


class Dtype:
    Bool: ClassVar[Dtype]
    Float32: ClassVar[Dtype]
    Float64: ClassVar[Dtype]
    Int16: ClassVar[Dtype]
    Int32: ClassVar[Dtype]
    Int64: ClassVar[Dtype]
    Int8: ClassVar[Dtype]
    UInt16: ClassVar[Dtype]
    UInt32: ClassVar[Dtype]
    UInt64: ClassVar[Dtype]
    UInt8: ClassVar[Dtype]
    Undefined: ClassVar[Dtype]
    def __init__(self, dtype_code: DtypeCode, byte_size: int, name: str) -> None: ...
    def byte_code(self) -> DtypeCode: ...
    def byte_size(self) -> int: ...


# bool = Dtype.Bool, this variable conflict with the builtin bool
bool8 = Dtype.Bool
float32 = Dtype.Float32
float64 = Dtype.Float64
int8 = Dtype.Int8
int16 = Dtype.Int16
int32 = Dtype.Int32
int64 = Dtype.Int64
uint8 = Dtype.UInt8
uint16 = Dtype.UInt16
uint32 = Dtype.UInt32
uint64 = Dtype.UInt64
undefined = Dtype.Undefined


class DynamicSizeVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: DynamicSizeVector) -> None: ...
    @overload
    def __init__(self, dim_sizes: Iterable) -> None: ...
    def append(self, x: Optional[int]) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: Optional[int]) -> int: ...
    def extend(self, L: Union[DynamicSizeVector, Iterable]) -> None: ...
    def insert(self, i: int, x: Optional[int]) -> None: ...
    def pop(self, i: Optional[int] = None) -> Optional[int]: ...
    def remove(self, x: Optional[int]) -> None: ...


class SizeVector:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: SizeVector) -> None: ...
    @overload
    def __init__(self, dim_sizes: Iterable) -> None: ...
    def append(self, x: int) -> None: ...
    def clear(self) -> None: ...
    def count(self, x: int) -> int: ...
    def extend(self, L: Union[SizeVector, Iterable]) -> None: ...
    def insert(self, i: int, x: int) -> None: ...
    def pop(self, i: Optional[int] = None) -> int: ...
    def remove(self, x: int) -> None: ...


class HashMap:
    @overload
    def __init__(
        self,
        init_capacity: int,
        key_dtype: Dtype,
        key_element_shape: Iterable,
        value_dtype: Dtype,
        value_element_shape: Iterable,
        device: Device = Device("CPU:0"),
    ) -> None: ...

    @overload
    def __init__(
        self,
        init_capacity: int,
        key_dtype: Dtype,
        key_element_shape: Iterable,
        value_dtypes: Sequence[Dtype],
        value_element_shapes: Sequence[Iterable],
        device: Device = Device("CPU:0"),
    ) -> None: ...
    def activate(self, keys: Tensor) -> Tuple[Tensor, Tensor]: ...
    def active_buf_indices(self) -> Tensor: ...
    def capacity(self) -> int: ...
    def clone(self) -> HashMap: ...
    def cpu(self) -> HashMap: ...
    def cuda(self, device_id: int = 0) -> HashMap: ...
    def erase(self, keys: Tensor) -> Tensor: ...
    def find(self, keys: Tensor) -> Tuple[Tensor, Tensor]: ...
    @overload
    def insert(self, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]: ...
    @overload
    def insert(self, keys: Tensor, list_values: Sequence[Tensor]) -> Tuple[Tensor, Tensor]: ...
    def key_tensor(self) -> Tensor: ...
    @classmethod
    def load(cls, file_name: str) -> HashMap: ...
    def reserve(self, capacity: int) -> None: ...
    def save(self, file_name: str) -> None: ...
    def size(self) -> int: ...
    def to(self, device: Device, copy: bool = False) -> HashMap: ...
    @overload
    def value_tensor(self) -> Tensor: ...
    @overload
    def value_tensor(self, value_buffer_id: int) -> Tensor: ...
    def value_tensors(self) -> List[Tensor]: ...


class HashSet:
    def __init__(
        self,
        init_capacity: int,
        key_dtype: Dtype,
        key_element_shape: SizeVector,
        device: Device = Device("CPU:0"),
    ) -> None: ...
    def active_buf_indices(self) -> Tensor: ...
    def capacity(self) -> int: ...
    def clone(self) -> HashSet: ...
    def cpu(self) -> HashSet: ...
    def cuda(self, device_id: int = 0) -> HashSet: ...
    def erase(self, keys: Tensor) -> Tensor: ...
    def find(self, keys: Tensor) -> Tuple[Tensor, Tensor]: ...
    def insert(self, keys: Tensor) -> Tuple[Tensor, Tensor]: ...
    def key_tensor(self) -> Tensor: ...
    @classmethod
    def load(cls, file_name: str) -> HashSet: ...
    def reserve(self, capacity: int) -> None: ...
    def save(self, file_name: str) -> None: ...
    def size(self) -> int: ...
    def to(self, device: Device, copy: bool = False) -> HashSet: ...


class Scalar:
    def __init__(self, v: Union[float, int, bool]) -> None: ...


class Tensor:
    @overload
    def __init__(
        self,
        np_array: ArrayLike,
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        scalar_value: Union[bool, int, float],
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        shape: Sequence[int],
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> None: ...
    def abs(self) -> Tensor: ...
    def abs_(self) -> Tensor: ...
    def add(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def add_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def __add__(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def all(self) -> bool: ...

    def allclose(
        self, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08
    ) -> bool: ...
    def any(self) -> bool: ...

    @overload
    @classmethod
    def arange(
        cls,
        stop: Union[int, float],
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> Tensor: ...

    @overload
    @classmethod
    def arange(
        cls,
        start: Union[int, float, None],
        stop: Union[int, float, None],
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> Tensor: ...
    def argmax(self, dim: Optional[SizeVector] = None) -> Tensor: ...
    def argmin(self, dim: Optional[SizeVector] = None) -> Tensor: ...
    def ceil(self) -> Tensor: ...
    def reshape(self, dst_shape: SizeVector | ArrayLike) -> Tensor: ...
    def shape(self) -> SizeVector: ...
    def clip(self, min_val: Scalar, max_val: Scalar) -> Tensor: ...
    def clip_(self, min_val: Scalar, max_val: Scalar) -> Tensor: ...
    def clone(self) -> Tensor: ...
    def contiguous(self) -> Tensor: ...
    def cos(self) -> Tensor: ...
    def cos_(self) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def cuda(self, device_id: int = 0) -> Tensor: ...
    def det(self) -> float: ...
    @classmethod
    def diag(cls, input: Tensor) -> Tensor: ...
    def div(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def div_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def __sub__(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def __mul__(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def __div__(self, value: Union[Scalar, Tensor]) -> Tensor: ...

    @classmethod
    def empty(
        cls,
        shape: Optional[SizeVector] = None,
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> Tensor: ...
    def eq(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def eq_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def exp(self) -> Tensor: ...
    def exp_(self) -> Tensor: ...

    @classmethod
    def eye(
        cls, n: int, dtype: Optional[Dtype] = None, device: Optional[Device] = None
    ) -> Tensor: ...
    def floor(self) -> Tensor: ...
    @classmethod
    def from_dlpack(cls, dlmt) -> Tensor: ...
    @classmethod
    def from_numpy(cls, array: ndarray) -> Tensor: ...

    @classmethod
    def full(
        cls,
        shape: Union[Tuple, List, SizeVector],
        fill_value: Union[float, int, bool],
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
    ) -> Tensor: ...
    def ge(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def ge_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def gt(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def gt_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def inv(self) -> Tensor: ...
    def is_contiguous(self) -> bool: ...

    def isclose(
        self, other: Tensor, rtol: float = 1e-05, atol: float = 1e-08
    ) -> Tensor: ...
    def isfinite(self) -> Tensor: ...
    def isinf(self) -> Tensor: ...
    def isnan(self) -> Tensor: ...
    def issame(self, other: Tensor) -> bool: ...
    def item(self) -> Scalar: ...
    def le(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def le_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    @classmethod
    def load(cls, file_name: str) -> Tensor: ...
    def logical_and(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def logical_and_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def logical_not(self) -> Tensor: ...
    def logical_not_(self) -> Tensor: ...
    def logical_or(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def logical_or_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def logical_xor(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def logical_xor_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def lstsq(self, B: Tensor) -> Tensor: ...
    def lt(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def lt_(self, value: Union[Scalar, Tensor]) -> Tensor: ...
    def lu(self, perumute_l: bool = False) -> Tuple[Tensor, Tensor, Tensor]: ...
    def lu_ipiv(self) -> Tuple[Tensor, Tensor]: ...
    def matmul(self, rhs: Tensor) -> Tensor: ...
    def max(self, dim: Optional[SizeVector] = None, keepdim: bool = False) -> Tensor: ...
    def mean(self, dim: Optional[SizeVector] = None, keepdim: bool = False) -> Tensor: ...
    def min(self, dim: Optional[SizeVector] = None, keepdim: bool = False) -> Tensor: ...
    @overload
    def to(self, dtype: Dtype, copy: bool = False) -> Tensor: ...
    @overload
    def to(self, device: Device, copy: bool = False) -> Tensor: ...
    @overload
    def __getitem__(self, indices: Tensor) -> Tensor: ...
    @overload
    def __getitem__(self, indices: Tuple[slice, slice]) -> Tensor: ...


def addmm(input: Tensor, A: Tensor, B: Tensor, alpha: float, beta: float) -> Tensor: ...
def append(self: Tensor, values: Tensor, axis: Optional[int] = None) -> Tensor: ...
def concatenate(tensors: Sequence[Tensor], axis: Optional[int] = None) -> Tensor: ...
def det(A: Tensor) -> float: ...
def inv(A: Tensor) -> Tensor: ...
def lstsq(A: Tensor, B: Tensor) -> Tensor: ...
def lu(A: Tensor, permute_l: bool = False) -> Tuple[Tensor, Tensor]: ...
def lu_ipiv(A: Tensor) -> Tuple[Tensor, Tensor]: ...
def matmul(A: Tensor, B: Tensor) -> Tensor: ...
def solve(A: Tensor, B: Tensor) -> Tensor: ...
def svd(A: Tensor) -> Tuple[Tensor, Tensor, Tensor]: ...
def tril(A: Tensor, diagonal: int = 0) -> Tensor: ...
def triu(A: Tensor, diagonal: int = 0) -> Tensor: ...
def triul(A: Tensor, diagonal: int = 0) -> Tuple[Tensor, Tensor]: ...
