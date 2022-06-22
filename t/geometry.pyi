from __future__ import annotations

from typing import Iterable, Optional, Sequence, overload, Tuple, Dict, Iterator, Union
from enum import Enum
from numpy import Inf
from numpy.typing import ArrayLike

import open3d

from .. import core
from .. import geometry


class Geometry:
    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Geometry: ...
    def is_empty(self) -> bool: ...


class Image(Geometry):
    channels: int
    columns: int
    device: core.Device
    dtype: core.Dtype
    rows: int

    @overload
    def __init__(
        self,
        rows: int = 0,
        cols: int = 0,
        channels: int = 1,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> None: ...
    @overload
    def __init__(self, tensor: core.Tensor) -> None: ...
    def as_tensor(self) -> core.Tensor: ...
    def clear(self) -> Image: ...

    def clip_transform(
        self, scale: float, min_value: float, max_value: float, clip_fill: float = 0.0
    ) -> Image: ...
    def clone(self) -> Image: ...

    def colorize_depth(
        self, scale: float, min_value: float, max_value: float
    ) -> Image: ...
    def cpu(self) -> Image: ...
    def create_normal_map(self, invalid_fill: float = 0.0) -> Image: ...

    def create_vertex_map(
        self, intrinsics: core.Tensor, invalid_fill: float = 0.0
    ) -> Image: ...
    def cuda(self, device_id: int = 0) -> Image: ...
    def dilate(self, kernel_size: int = 3) -> Image: ...
    def filter(self, kernel: core.Tensor) -> Image: ...

    def filter_bilateral(
        self, kernel_size: int = 3, value_sigma: float = 20.0, dist_sigma: float = 10.0
    ) -> Image: ...
    def filter_gaussian(self, kernel_size: int = 3, sigma: float = 1.0) -> Image: ...
    def filter_sobel(self, kernel_size: int = 3) -> Tuple[Image, Image]: ...

    @classmethod
    def from_legacy_image(
        cls, image_legacy: geometry.Image, device: core.Device = core.Device("CPU:0")
    ) -> Image: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def linear_transform(self, scale: float = 1.0, offset: float = 0.0) -> Image: ...
    def pyrdown(self) -> Image: ...

    def resize(
        self, sampling_rate: float = 0.5, interp_type: InterpType = InterpType.Nearest
    ) -> Image: ...
    def rgb_to_gray(self) -> Image: ...
    @overload
    def to(self, device: core.Device, copy: bool = False) -> Image: ...

    @overload
    def to(
        self,
        dtype: core.Dtype,
        scale: Optional[float] = None,
        offset: float = 0.0,
        copy: bool = False,
    ) -> Image: ...
    def to_legacy_image(self) -> geometry.Image: ...


class InterpType(Enum):
    Cubic = ...
    Lanczos = ...
    Linear = ...
    Nearest = ...
    Super = ...


class PointCloud(Geometry):
    point: TensorMap
    @overload
    def __init__(self, device: core.Device) -> None: ...
    @overload
    def __init__(self, points: core.Tensor) -> None: ...
    @overload
    def __init__(self, map_keys_to_tensors: Dict[str, core.Tensor]) -> None: ...
    def append(self, other: PointCloud) -> PointCloud: ...
    def clone(self) -> PointCloud: ...
    def cpu(self) -> PointCloud: ...

    @classmethod
    def create_from_depth_image(
        cls,
        depth: Image,
        intrinsics: core.Tensor,
        extrinsics: core.Tensor = ...,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        stride: int = 1,
        with_normals: bool = False,
    ) -> PointCloud: ...

    @classmethod
    def create_from_rgbd_image(
        cls,
        rgbd_image: RGBDImage,
        intrinsics: core.Tensor,
        extrinsics: core.Tensor = ...,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        stride: int = 1,
        with_normals: bool = False,
    ) -> PointCloud: ...
    def cuda(self, device_id: int = 0) -> PointCloud: ...

    @classmethod
    def from_legacy(
        cls,
        pcd_legacy: geometry.PointCloud,
        dtype: core.Dtype = core.Dtype.Float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> PointCloud: ...
    def get_center(self) -> core.Tensor: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def rotate(self, R: core.Tensor, center: core.Tensor) -> PointCloud: ...
    def scale(self, scale: float, center: core.Tensor) -> PointCloud: ...
    def to(self, device: core.Device, copy: bool = False) -> PointCloud: ...
    def to_legacy(self) -> geometry.PointCloud: ...
    def transform(self, transformation: core.Tensor) -> PointCloud: ...

    def translate(
        self, translation: core.Tensor, relative: bool = True
    ) -> PointCloud: ...
    def voxel_down_sample(self, voxel_size: float) -> PointCloud: ...


class RGBDImage(Geometry):
    aligned_: bool
    color: Image
    depth: Image
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, color: Image, depth: Image, aligned: bool = True) -> None: ...
    def are_aligned(self) -> bool: ...
    def clear(self) -> RGBDImage: ...
    def clone(self) -> RGBDImage: ...
    def cpu(self) -> RGBDImage: ...
    def cuda(self, device_id: int = 0) -> RGBDImage: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def to(self, device: core.Device, copy: bool = False) -> RGBDImage: ...
    def to_legacy_rgbd_image(self) -> geometry.RGBDImage: ...


class SurfaceMaskCode(Enum):
    ColorMap = ...
    DepthMap = ...
    NormalMap = ...
    VertexMap = ...


class TensorMap:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, primary_key: str) -> None: ...

    @overload
    def __init__(
        self, primary_key: str, map_keys_to_tensors: Dict[str, core.Tensor]
    ) -> None: ...
    def assert_size_synchronized(self) -> None: ...
    def erase(self, key: str) -> int: ...
    def get_primary_key(self) -> str: ...
    def is_size_synchronized(self) -> bool: ...
    def items(self) -> Iterator: ...
    def __getitem__(self, key: str) -> core.Tensor: ...
    def __setitem__(self, key: str, value: core.Tensor) -> TensorMap: ...


class TriangleMesh(Geometry):
    triangle: TensorMap
    vertex: TensorMap
    @overload
    def __init__(self, device: core.Device = core.Device("CPU:0")) -> None: ...

    @overload
    def __init__(
        self, vertex_positions: core.Tensor, triangle_indices: core.Tensor
    ) -> None: ...
    def clear(self) -> TriangleMesh: ...
    def clone(self) -> TriangleMesh: ...
    def cpu(self) -> TriangleMesh: ...
    def cuda(self, device_id: int = 0) -> TriangleMesh: ...

    @classmethod
    def from_legacy(
        cls,
        mesh_legacy: geometry.TriangleMesh,
        vertex_dtype: core.Dtype = core.float32,
        triangle_dtype: core.Dtype = core.int64,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    def get_center(self) -> core.Tensor: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def has_valid_material(self) -> bool: ...
    def rotate(self, R: core.Tensor, center: core.Tensor) -> TriangleMesh: ...
    def scale(self, scale: float, center: core.Tensor) -> TriangleMesh: ...
    def to(self, device: core.Device, copy: bool = False) -> TriangleMesh: ...
    def to_legacy(self) -> geometry.TriangleMesh: ...
    def transform(self, transformation: core.Tensor) -> TriangleMesh: ...

    def translate(
        self, translation: core.Tensor, relative: bool = True
    ) -> TriangleMesh: ...


class VoxelBlockGrid:
    def __init__(
        self,
        attr_names: Sequence[str],
        attr_dtypes: Sequence[core.Dtype],
        attr_channels: Sequence[Union[Iterable, int]],
        voxel_size: float = 0.0058,
        block_resolution: int = 16,
        block_count: int = 10000,
        device: core.Device = core.Device("CPU:0"),
    ) -> None: ...
    def attribute(self, attribute_name: str) -> core.Tensor: ...

    @overload
    def compute_unique_block_coordinates(
        self,
        depth: Image,
        intrisic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> core.Tensor: ...

    @overload
    def compute_unique_block_coordinates(
        self,
        pcd: PointCloud,
        trunc_voxel_multiplier: float = 8.0,
    ) -> core.Tensor: ...

    def extract_point_cloud(
        self, weight_threshold: float = 3.0, estimated_point_number: int = -1
    ) -> PointCloud: ...

    def extract_triangle_mesh(
        self, weight_threshold: float = 3.0, estimated_point_number: int = -1
    ) -> TriangleMesh: ...
    def hashmap(self) -> core.HashMap: ...

    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        color: Image,
        depth_intrinsic: core.Tensor,
        color_intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...

    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        color: Image,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...

    @overload
    def integrate(
        self,
        block_coords: core.Tensor,
        depth: Image,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
    ) -> None: ...
    @classmethod
    def load(cls, file_name: str) -> VoxelBlockGrid: ...

    def ray_cast(
        self,
        block_coords: core.Tensor,
        intrinsic: core.Tensor,
        extrinsic: core.Tensor,
        width: int,
        height: int,
        render_attributes: Sequence[str] = ["depth", "color"],
        depth_scale: float = 1000.0,
        depth_min: float = 0.1,
        depth_max: float = 3.0,
        weight_threshold: float = 3.0,
        trunc_voxel_multiplier: float = 8.0,
        range_map_down_factor: int = 8,
    ) -> TensorMap: ...
    def save(self, file_name: str) -> None: ...
    def voxel_coordinates(self, voxel_indices: core.Tensor) -> core.Tensor: ...

    @overload
    def voxel_coordinates_and_flattened_indices(
        self, buf_indices: core.Tensor
    ) -> Tuple[core.Tensor, core.Tensor]: ...

    @overload
    def voxel_coordinates_and_flattened_indices(
        self,
    ) -> Tuple[core.Tensor, core.Tensor]: ...
    @overload
    def voxel_indices(self, buf_indices: core.Tensor) -> core.Tensor: ...
    @overload
    def voxel_indices(self) -> core.Tensor: ...


class RaycastingScene:
    @overload
    def add_triangles(self, mesh: TriangleMesh) -> int:
        """
        Add a triangle mesh to the scene.

        Args:
            mesh (open3d.t.geometry.TriangleMesh): A triangle mesh.

        Returns:
            The geometry ID of the added mesh.
        """
        ...

    @overload
    def add_triangles(self, vertex_positions: core.Tensor, triangle_indices: core.Tensor) -> int:
        """
        Add a triangle mesh to the scene.

        Args:
            vertices (open3d.core.Tensor): Vertices as Tensor of dim {N,3} and dtype Float32.
            triangles (open3d.core.Tensor): Triangles as Tensor of dim {M,3} and dtype UInt32.

        Returns:
            The geometry ID of the added mesh.
        """
        ...

    def cast_rays(self, rays: core.Tensor, nthreads: int = 0) -> dict[str, core.Tensor]:
        """
        Computes the first intersection of the rays with the scene.

        Args:
            rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction but the returned hit distance uses the length of the direction vector as unit.
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A dictionary which contains the following keys
            t_hit: A tensor with the distance to the first hit. The shape is {..}. If there is no intersection the hit distance is inf.
            geometry_ids: A tensor with the geometry IDs. The shape is {..}. If there is no intersection the ID is INVALID_ID.
            primitive_ids: A tensor with the primitive IDs, which corresponds to the triangle index. The shape is {..}. If there is no intersection the ID is INVALID_ID.
            primitive_uvs: A tensor with the barycentric coordinates of the hit points within the hit triangles. The shape is {.., 2}.
            primitive_normals: A tensor with the normals of the hit triangles. The shape is {.., 3}.
        """
        ...

    def compute_closest_points(self, query_points: core.Tensor, nthreads: int = 0) -> dict[str, core.Tensor]:
        """
        Computes the closest points on the surfaces of the scene.

        Args:
            query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query_point to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            The returned dictionary contains
            points: A tensor with the closest surface points. The shape is {..}.
            geometry_ids: A tensor with the geometry IDs. The shape is {..}.
            primitive_ids: A tensor with the primitive IDs, which corresponds to the triangle index. The shape is {..}.
        """
        ...

    def compute_distance(self, query_points: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """
        Computes the distance to the surface of the scene.

        Args:
            query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query_point to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A tensor with the distances to the surface. The shape is {..}.
        """
        ...

    def compute_occupancy(self, query_points: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """
        Computes the occupancy at the query point positions.

        This function computes whether the query points are inside or outside. The function assumes that all meshes are
        watertight and that there are no intersections between meshes, i.e., inside and outside must be well defined. 
        The function determines if a point is inside by counting the intersections of a rays starting at the query points.

        Args:
            query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query_point to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A tensor with the distances to the surface. The shape is {..}.
        """
        ...

    def compute_signed_distance(self, query_points: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """
        Computes the signed distance to the surface of the scene.

        This function computes the signed distance to the meshes in the scene. The function assumes that all meshes are
        watertight and that there are no intersections between meshes, i.e., inside and outside must be well defined.
        The function determines the sign of the distance by counting the intersections of a rays starting at the query points.

        Args:
            query_points (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 3}, and Dtype Float32 describing the query points. {..} can be any number of dimensions, e.g., to organize the query_point to create a 3D grid the shape can be {depth, height, width, 3}. The last dimension must be 3 and has the format [x, y, z].
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A tensor with the signed distances to the surface. The shape is {..}. Negative distances mean a point is inside a closed surface.
        """
        ...

    def count_intersections(self, rays: core.Tensor, nthreads: int = 0) -> core.Tensor:
        """
        Computes the number of intersection of the rays with the scene.

        Args:
            rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction.
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A tensor with the number of intersections. The shape is {..}.
        """
        ...

    def test_occlusions(self, rays: core.Tensor, tnear: float = 0.0,
                        tfar: float = Inf, nthreads: int = 0) -> core.Tensor:
        """
        Checks if the rays have any intersection with the scene.

        Args:
            rays (open3d.core.Tensor): A tensor with >=2 dims, shape {.., 6}, and Dtype Float32 describing the rays. {..} can be any number of dimensions, e.g., to organize rays for creating an image the shape can be {height, width, 6}. The last dimension must be 6 and has the format [ox, oy, oz, dx, dy, dz] with [ox,oy,oz] as the origin and [dx,dy,dz] as the direction. It is not necessary to normalize the direction.
            tnear (float): The tnear offset for the rays. The default is 0.
            tfar (float): The tfar value for the ray. The default is infinity.
            nthreads (int): The number of threads to use. Set to 0 for automatic.

        Returns:
            A boolean tensor which indicates if the ray is occluded by the scene (true) or not (false).
        """
        ...

    @overload
    @staticmethod
    def create_rays_pinhole(intrinsic_matrix: core.Tensor, extrinsic_matrix: core.Tensor,
                            width_px: int, height_px: int) -> core.Tensor:
        """
        Creates rays for the given camera parameters.

        Args:
            intrinsic_matrix (open3d.core.Tensor): The upper triangular intrinsic matrix with shape {3,3}.
            extrinsic_matrix (open3d.core.Tensor): The 4x4 world to camera SE(3) transformation matrix.
            width_px (int): The width of the image in pixels.
            height_px (int): The height of the image in pixels.

        Returns:
            A tensor of shape {height_px, width_px, 6} with the rays.
        """
        ...

    @overload
    @staticmethod
    def create_rays_pinhole(fov_deg: float, center: core.Tensor | ArrayLike, eye: core.Tensor | ArrayLike,
                            up: core.Tensor | ArrayLike, width_px: int, height_px: int) -> core.Tensor:
        """
        Creates rays for the given camera parameters.

        Args:
            fov_deg (float): The horizontal field of view in degree.
            center (open3d.core.Tensor): The point the camera is looking at with shape {3}.
            eye (open3d.core.Tensor): The position of the camera with shape {3}.
            up (open3d.core.Tensor): The up-vector with shape {3}.
            width_px (int): The width of the image in pixels.
            height_px (int): The height of the image in pixels.

        Returns:
            A tensor of shape {height_px, width_px, 6} with the rays.
        """
        ...
