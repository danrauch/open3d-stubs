from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, overload, Tuple, Dict, Iterator, Union
from enum import Enum
import typing
from core.core import Dtype
import numpy as np
from numpy import Inf
from numpy.typing import ArrayLike

import open3d

from .. import core
from .. import geometry

class DrawableGeometry:
    """Base class for geometry types which can be visualized."""

    def has_valid_material(self) -> bool:
        """Returns true if the geometry's material is valid."""
    @property
    def material(self) -> open3d.visualization.rendering.Material: ...
    @material.setter
    def material(self, material: open3d.visualization.rendering.Material) -> None: ...

class Geometry:
    """The base geometry class."""

    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Geometry: ...
    def is_empty(self) -> bool: ...
    @property
    def device(self) -> core.Device:
        """
        Returns the device of the geometry.

        Args:
            core.Device
        """
    @property
    def is_cpu(self) -> bool:
        """Returns true if the geometry is on CPU."""
    @property
    def is_cuda(self) -> bool:
        """Returns true if the geometry is on CUDA."""

class AxisAlignedBoundingBox(Geometry, DrawableGeometry):
    """
    A bounding box that is aligned along the coordinate axes
    and defined by the min_bound and max_bound.
    - (min_bound, max_bound): Lower and upper bounds of the bounding box for all axes.
        - Usage
            - AxisAlignedBoundingBox::GetMinBound()
            - AxisAlignedBoundingBox::SetMinBound(const core::Tensor &min_bound)
            - AxisAlignedBoundingBox::GetMaxBound()
            - AxisAlignedBoundingBox::SetMaxBound(const core::Tensor &max_bound)
        - Value tensor must have shape {3,}.
        - Value tensor must have the same data type and device.
        - Value tensor can only be float32 (default) or float64.
        - The device of the tensor determines the device of the box.

    - color: Color of the bounding box.
        - Usage
            - AxisAlignedBoundingBox::GetColor()
            - AxisAlignedBoundingBox::SetColor(const core::Tensor &color)
        - Value tensor must have shape {3,}.
        - Value tensor can only be float32 (default) or float64.
        - Value tensor can only be range [0.0, 1.0].
    """

    def __add__(self, other: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """
        Add operation for axis-aligned bounding box.
        The device of ohter box must be the same as the device of the current box.
        """
    def __repr__(self) -> str: ...
    def clone(self) -> AxisAlignedBoundingBox:
        """Returns copy of the axis-aligned box on the same device."""
    def cpu(self) -> AxisAlignedBoundingBox:
        """
        Transfer the axis-aligned box to CPU. If the axis-aligned box is already on CPU, no copy will be performed.
        """
    def cuda(self, device_id: int = 0) -> AxisAlignedBoundingBox:
        """
        Transfer the axis-aligned box to a CUDA device. If the axis-aligned box is already on the specified CUDA device,
        no copy will be performed.
        """
    @staticmethod
    def from_legacy(
        aabb: geometry.AxisAlignedBoundingBox,
        dtype: core.Dtype = core.float32,
        device: core.Device = core.Device("CPU:0"),
    ) -> AxisAlignedBoundingBox:
        """Create an AxisAlignedBoundingBox from a legacy Open3D axis-aligned box."""
    @staticmethod
    def create_from_points(
        points: core.Tensor,
    ) -> AxisAlignedBoundingBox:
        """Creates the axis-aligned box that encloses the set of points."""
    def get_box_points(self) -> core.Tensor:
        """
        Returns the eight points that define the bounding box. The Return tensor has shape {8, 3}
        and data type of float32.
        """
    def get_center(self) -> core.Tensor:
        """Returns the center for box coordinates."""
    def get_color(self) -> core.Tensor:
        """Returns the color for box."""
    def get_extent(self) -> core.Tensor:
        """Get the extent/length of the bounding box in x, y, and z dimension."""
    def get_half_extent(self) -> core.Tensor:
        """Returns the half extent of the bounding box."""
    def get_max_bound(self) -> core.Tensor:
        """Returns the max bound for box coordinates."""
    def get_max_extent(self) -> float:
        """Returns the maximum extent, i.e. the maximum of X, Y and Z axis's extents."""
    def get_min_bound(self) -> core.Tensor:
        """Returns the min bound for box coordinates."""
    def to(self, device: core.Device, copy: bool = False) -> AxisAlignedBoundingBox:
        """Transfer the axis-aligned box to a specified device."""
    def to_legacy(self) -> open3d.cpu.pybind.geometry.AxisAlignedBoundingBox:
        """Convert to a legacy Open3D axis-aligned box."""
    def volume(self) -> float:
        """Returns the volume of the bounding box."""

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
    def clip_transform(self, scale: float, min_value: float, max_value: float, clip_fill: float = 0.0) -> Image: ...
    def clone(self) -> Image: ...
    def colorize_depth(self, scale: float, min_value: float, max_value: float) -> Image: ...
    def cpu(self) -> Image: ...
    def create_normal_map(self, invalid_fill: float = 0.0) -> Image: ...
    def create_vertex_map(self, intrinsics: core.Tensor, invalid_fill: float = 0.0) -> Image: ...
    def cuda(self, device_id: int = 0) -> Image: ...
    def dilate(self, kernel_size: int = 3) -> Image: ...
    def filter(self, kernel: core.Tensor) -> Image: ...
    def filter_bilateral(self, kernel_size: int = 3, value_sigma: float = 20.0, dist_sigma: float = 10.0) -> Image: ...
    def filter_gaussian(self, kernel_size: int = 3, sigma: float = 1.0) -> Image: ...
    def filter_sobel(self, kernel_size: int = 3) -> Tuple[Image, Image]: ...
    @classmethod
    def from_legacy_image(cls, image_legacy: geometry.Image, device: core.Device = core.Device("CPU:0")) -> Image: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def linear_transform(self, scale: float = 1.0, offset: float = 0.0) -> Image: ...
    def pyrdown(self) -> Image: ...
    def resize(self, sampling_rate: float = 0.5, interp_type: InterpType = InterpType.Nearest) -> Image: ...
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

class LineSet(Geometry, DrawableGeometry):
    """
    A LineSet contains points and lines joining them and optionally attributes on
    the points and lines.  The ``LineSet`` class stores the attribute data in
    key-value maps, where the key is the attribute name and value is a Tensor
    containing the attribute data.  There are two maps: one each for ``point``
    and ``line``.

    The attributes of the line set have different levels::

        import open3d as o3d

        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32

        # Create an empty line set
        # Use lineset.point to access the point attributes
        # Use lineset.line to access the line attributes
        lineset = o3d.t.geometry.LineSet()

        # Default attribute: point.positions, line.indices
        # These attributes is created by default and are required by all line
        # sets. The shape must be (N, 3) and (N, 2) respectively. The device of
        # "positions" determines the device of the line set.
        lineset.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                      [0, 0, 1],
                                                      [0, 1, 0],
                                                      [0, 1, 1]], dtype_f, device)
        lineset.line.indices = o3d.core.Tensor([[0, 1],
                                                   [1, 2],
                                                   [2, 3],
                                                   [3, 0]], dtype_i, device)

        # Common attributes: line.colors
        # Common attributes are used in built-in line set operations. The
        # spellings must be correct. For example, if "color" is used instead of
        # "color", some internal operations that expects "colors" will not work.
        # "colors" must have shape (N, 3) and must be on the same device as the
        # line set.
        lineset.line.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                                  [0.1, 0.1, 0.1],
                                                  [0.2, 0.2, 0.2],
                                                  [0.3, 0.3, 0.3]], dtype_f, device)

        # User-defined attributes
        # You can also attach custom attributes. The value tensor must be on the
        # same device as the line set. The are no restrictions on the shape or
        # dtype, e.g.,
        lineset.point.labels = o3d.core.Tensor(...)
        lineset.line.features = o3d.core.Tensor(...)
    """

    def __getstate__(self) -> tuple: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def clone(self) -> LineSet:
        """
        Returns copy of the line set on the same device.
        """
    def cpu(self) -> LineSet:
        """
        Transfer the line set to CPU. If the line set is already on CPU, no copy will be performed.
        """
    def cuda(self, device_id: int = 0) -> LineSet:
        """
        Transfer the line set to a CUDA device. If the line set is already on the specified CUDA device, no copy will be performed.
        """
    @staticmethod
    def extrude_linear(*args, **kwargs) -> typing.Any:
        """
        Sweeps the line set along a direction vector.

        Args:

            vector (open3d.core.Tensor): The direction vector.

            scale (float): Scalar factor which essentially scales the direction vector.

        Returns:
            A triangle mesh with the result of the sweep operation.


        Example:

            This code generates an L-shaped mesh::
                import open3d as o3d

                lines = o3d.t.geometry.LineSet([[1.0,0.0,0.0],[0,0,0],[0,0,1]], [[0,1],[1,2]])
                mesh = lines.extrude_linear([0,1,0])
                o3d.visualization.draw([{'name': 'L', 'geometry': mesh}])
        """
    @staticmethod
    def extrude_rotation(*args, **kwargs) -> typing.Any:
        """
        Sweeps the line set rotationally about an axis.

        Args:
            angle (float): The rotation angle in degree.

            axis (open3d.core.Tensor): The rotation axis.

            resolution (int): The resolution defines the number of intermediate sweeps
                about the rotation axis.

            translation (float): The translation along the rotation axis.

        Returns:
            A triangle mesh with the result of the sweep operation.


        Example:

            This code generates a spring from a single line::

                import open3d as o3d

                line = o3d.t.geometry.LineSet([[0.7,0,0],[1,0,0]], [[0,1]])
                spring = line.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
                o3d.visualization.draw([{'name': 'spring', 'geometry': spring}])
        """
    @staticmethod
    def get_axis_aligned_bounding_box(*args, **kwargs) -> typing.Any:
        """
        Create an axis-aligned bounding box from point attribute 'positions'.
        """
    def get_center(self) -> core.Tensor:
        """
        Returns the center for point coordinates.
        """
    def get_max_bound(self) -> core.Tensor:
        """
        Returns the max bound for point coordinates.
        """
    def get_min_bound(self) -> core.Tensor:
        """
        Returns the min bound for point coordinates.
        """
    def to(self, device: core.Device, copy: bool = False) -> LineSet:
        """
        Transfer the line set to a specified device.
        """
    def to_legacy(self) -> open3d.cpu.pybind.geometry.LineSet:
        """
        Convert to a legacy Open3D LineSet.
        """
    @property
    def line(self) -> TensorMap:
        """
        Dictionary containing line attributes. The primary key ``indices`` contains indices of points defining the lines.

        :type: TensorMap
        """
    @property
    def point(self) -> TensorMap:
        """
        Dictionary containing point attributes. The primary key ``positions`` contains point positions.

        :type: TensorMap
        """

class TensorMap:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, primary_key: str) -> None: ...
    @overload
    def __init__(self, primary_key: str, map_keys_to_tensors: Dict[str, core.Tensor]) -> None: ...
    def assert_size_synchronized(self) -> None: ...
    def erase(self, key: str) -> int: ...
    def get_primary_key(self) -> str: ...
    def is_size_synchronized(self) -> bool: ...
    def items(self) -> Iterator: ...
    def __getitem__(self, key: str) -> core.Tensor: ...
    def __setitem__(self, key: str, value: core.Tensor) -> TensorMap: ...

class PointCloud(Geometry, DrawableGeometry):
    """
    A point cloud contains a list of 3D points. The point cloud class stores the
    attribute data in key-value maps, where the key is a string representing the
    attribute name and the value is a Tensor containing the attribute data.

    The attributes of the point cloud have different levels::

        import open3d as o3d

        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32

        # Create an empty point cloud
        # Use pcd.point to access the points' attributes
        pcd = o3d.t.geometry.PointCloud(device)

        # Default attribute: "positions".
        # This attribute is created by default and is required by all point clouds.
        # The shape must be (N, 3). The device of "positions" determines the device
        # of the point cloud.
        pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                  [1, 1, 1],
                                                  [2, 2, 2]], dtype, device)

        # Common attributes: "normals", "colors".
        # Common attributes are used in built-in point cloud operations. The
        # spellings must be correct. For example, if "normal" is used instead of
        # "normals", some internal operations that expects "normals" will not work.
        # "normals" and "colors" must have shape (N, 3) and must be on the same
        # device as the point cloud.
        pcd.point.normals = o3d.core.Tensor([[0, 0, 1],
                                                [0, 1, 0],
                                                [1, 0, 0]], dtype, device)
        pcd.point.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                                [0.1, 0.1, 0.1],
                                                [0.2, 0.2, 0.2]], dtype, device)

        # User-defined attributes.
        # You can also attach custom attributes. The value tensor must be on the
        # same device as the point cloud. The are no restrictions on the shape and
        # dtype, e.g.,
        pcd.point.intensities = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
        pcd.point.labels = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)
    """

    def __add__(self, arg0: PointCloud) -> PointCloud: ...
    def __getstate__(self) -> tuple: ...
    @overload
    def __init__(self, device: core.Device) -> None: ...
    @overload
    def __init__(self, points: core.Tensor) -> None: ...
    @overload
    def __init__(self, map_keys_to_tensors: Dict[str, core.Tensor]) -> None: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def append(self, arg0: PointCloud) -> PointCloud: ...
    def clone(self) -> PointCloud:
        """
        Returns a copy of the point cloud on the same device.
        """
    def cluster_dbscan(self, eps: float, min_points: int, print_progress: bool = False) -> core.Tensor:
        """
        Cluster PointCloud using the DBSCAN algorithm  Ester et al.,'A
        Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
        with Noise', 1996. This is a wrapper for a CPU implementation and a copy of the
        point cloud data and resulting labels will be made.

        Args:
            eps. Density parameter that is used to find neighbouring points.
            min_points. Minimum number of points to form a cluster.
            print_progress (default False). If 'True' the progress is visualized in the console.

        Return:
            A Tensor list of point labels on the same device as the point cloud, -1
            indicates noise according to the algorithm.

        Example:
            We use Redwood dataset for demonstration::

                import matplotlib.pyplot as plt

                sample_ply_data = o3d.data.PLYPointCloud()
                pcd = o3d.t.io.read_point_cloud(sample_ply_data.path)
                labels = pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)

                max_label = labels.max().item()
                colors = plt.get_cmap("tab20")(
                        labels.numpy() / (max_label if max_label > 0 else 1))
                colors = o3d.core.Tensor(colors[:, :3], o3d.core.float32)
                colors[labels < 0] = 0
                pcd.point.colors = colors
                o3d.visualization.draw([pcd])
        """
    def compute_boundary_points(
        self, radius: float, max_nn: int = 30, angle_threshold: float = 90.0
    ) -> typing.Tuple[PointCloud, core.Tensor]:
        """
        Compute the boundary points of a point cloud.
        The implementation is inspired by the PCL implementation. Reference:
        https://pointclouds.org/documentation/classpcl_1_1_boundary_estimation.html

        Args:
            radius. Neighbor search radius parameter.
            max_nn (default 30). Maximum number of neighbors to search.
            angle_threshold (default 90.0). Angle threshold to decide if a point is on the boundary.

        Return:
            Tensor of boundary points and its boolean mask tensor.

        Example:
            We will load the DemoCropPointCloud dataset, compute its boundary points::

                ply_point_cloud = o3d.data.DemoCropPointCloud()
                pcd = o3d.t.io.read_point_cloud(ply_point_cloud.point_cloud_path)
                boundaries, mask = pcd.compute_boundary_points(radius, max_nn)
                boundaries.paint_uniform_color([1.0, 0.0, 0.0])
                o3d.visualization.draw([pcd, boundaries])

        """
    @staticmethod
    def compute_convex_hull(*args, **kwargs) -> typing.Any:
        """
        Compute the convex hull of a triangle mesh using qhull. This runs on the CPU.

        Args:
            joggle_inputs (default False). Handle precision problems by
            randomly perturbing the input data. Set to True if perturbing the input
            iis acceptable but you need convex simplicial output. If False,
            neighboring facets may be merged in case of precision problems. See
            `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle>`__ for more
            details.

        Return:
            TriangleMesh representing the convexh hull. This contains an
            extra vertex property "point_indices" that contains the index of the
            corresponding vertex in the original mesh.

        Example:
            We will load the Eagle dataset, compute and display it's convex hull::

                eagle = o3d.data.EaglePointCloud()
                pcd = o3d.t.io.read_point_cloud(eagle.path)
                hull = pcd.compute_convex_hull()
                o3d.visualization.draw([{'name': 'eagle', 'geometry': pcd}, {'name': 'convex hull', 'geometry': hull}])

        """
    def cpu(self) -> PointCloud:
        """
        Transfer the point cloud to CPU. If the point cloud is already on CPU, no copy will be performed.
        """
    def cuda(self, device_id: int = 0) -> PointCloud:
        """
        Transfer the point cloud to a CUDA device. If the point cloud is already on the specified CUDA device, no copy will be performed.
        """
    def estimate_color_gradients(
        self, max_nn: typing.Optional[int] = 30, radius: typing.Optional[float] = None
    ) -> None:
        """
        Function to estimate point color gradients. It uses KNN search (Not recommended to use on GPU) if only max_nn parameter is provided, Radius search (Not recommended to use on GPU) if only radius is provided and Hybrid Search (Recommended) if radius parameter is also provided.
        """
    @staticmethod
    def extrude_linear(*args, **kwargs) -> typing.Any:
        """
        Sweeps the point cloud along a direction vector.

        Args:

            vector (open3d.core.Tensor): The direction vector.

            scale (float): Scalar factor which essentially scales the direction vector.

        Returns:
            A line set with the result of the sweep operation.


        Example:

            This code generates a set of straight lines from a point cloud::
                import open3d as o3d
                import numpy as np
                pcd = o3d.t.geometry.PointCloud(np.random.rand(10,3))
                lines = pcd.extrude_linear([0,1,0])
                o3d.visualization.draw([{'name': 'lines', 'geometry': lines}])
        """
    @staticmethod
    def extrude_rotation(*args, **kwargs) -> typing.Any:
        """
        Sweeps the point set rotationally about an axis.

        Args:
            angle (float): The rotation angle in degree.

            axis (open3d.core.Tensor): The rotation axis.

            resolution (int): The resolution defines the number of intermediate sweeps
                about the rotation axis.

            translation (float): The translation along the rotation axis.

        Returns:
            A line set with the result of the sweep operation.


        Example:

            This code generates a number of helices from a point cloud::

                import open3d as o3d
                import numpy as np
                pcd = o3d.t.geometry.PointCloud(np.random.rand(10,3))
                helices = pcd.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
                o3d.visualization.draw([{'name': 'helices', 'geometry': helices}])
        """
    @staticmethod
    def from_legacy(*args, **kwargs) -> typing.Any:
        """
        Create a PointCloud from a legacy Open3D PointCloud.
        """
    def to_legacy(self) -> open3d.cpu.pybind.geometry.PointCloud:
        """
        Convert to a legacy Open3D PointCloud.
        """
    def transform(self, transformation: core.Tensor) -> PointCloud:
        """
        Transforms the points and normals (if exist).
        """
    def translate(self, translation: core.Tensor, relative: bool = True) -> PointCloud:
        """
        Translates points.
        """
    @property
    def point(self) -> TensorMap:
        """
        Point's attributes: positions, colors, normals, etc.

        :type: TensorMap
        """
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
    def voxel_down_sample(self, voxel_size: float) -> PointCloud:
        """Downsamples a point cloud with a specified voxel size."""
    def uniform_down_sample(self, every_k_points: int) -> PointCloud:
        """Downsamples a point cloud by selecting every kth index point and its attributes."""
    def random_down_sample(self, sampling_ration: float) -> PointCloud:
        """Downsample a pointcloud by selecting random index point and its attributes."""
    def farthest_point_down_sample(self, num_samples: int) -> PointCloud: ...
    @staticmethod
    def get_axis_aligned_bounding_box() -> AxisAlignedBoundingBox:
        """Create an axis-aligned bounding box from attribute "positions"."""
    def get_center(self) -> core.Tensor:
        """
        Returns the center for point coordinates.
        """
    def get_max_bound(self) -> core.Tensor:
        """
        Returns the max bound for point coordinates.
        """
    def get_min_bound(self) -> core.Tensor:
        """
        Returns the min bound for point coordinates.
        """
    @staticmethod
    def hidden_point_removal(*args, **kwargs) -> typing.Any:
        """
        Removes hidden points from a point cloud and returns a mesh of
        the remaining points. Based on Katz et al. 'Direct Visibility of Point Sets',
        2007. Additional information about the choice of radius for noisy point clouds
        can be found in Mehra et. al. 'Visibility of Noisy Point Cloud Data', 2010.
        This is a wrapper for a CPU implementation and a copy of the point cloud data
        and resulting visible triangle mesh and indiecs will be made.

        Args:
            camera_location. All points not visible from that location will be removed.
            radius. The radius of the spherical projection.

        Return:
            Tuple of visible triangle mesh and indices of visible points on the same
            device as the point cloud.

        Example:
            We use armadillo mesh to compute the visible points from given camera::

                # Convert mesh to a point cloud and estimate dimensions.
                armadillo_data = o3d.data.ArmadilloMesh()
                pcd = o3d.io.read_triangle_mesh(
                armadillo_data.path).sample_points_poisson_disk(5000)

                diameter = np.linalg.norm(
                        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

                # Define parameters used for hidden_point_removal.
                camera = o3d.core.Tensor([0, 0, diameter], o3d.core.float32)
                radius = diameter * 100

                # Get all points that are visible from given view point.
                pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
                _, pt_map = pcd.hidden_point_removal(camera, radius)
                pcd = pcd.select_by_index(pt_map)
                o3d.visualization.draw([pcd], point_size=5)
        """
    def remove_duplicated_points(self) -> typing.Tuple[PointCloud, core.Tensor]:
        """Remove duplicated points and there associated attributes."""
    def remove_radius_outliers(self, nb_points: int, search_radius: float) -> PointCloud:
        """Remove points that have less than nb_points neighbors in a sphere of a given search radius."""
    def remove_non_finite_points(
        self, remove_nan: bool = True, remove_infinite: bool = True
    ) -> typing.Tuple[PointCloud, core.Tensor]:
        """
        Remove all points from the point cloud that have a nan entry, or infinite value. It also removes the corresponding attributes.
        """
    def rotate(self, R: core.Tensor, center: core.Tensor) -> PointCloud:
        """
        Rotate points and normals (if exist).
        """
    def scale(self, scale: float, center: core.Tensor) -> PointCloud:
        """
        Scale points.
        """
    def segment_plane(
        self,
        distance_threshold: float = 0.01,
        ransac_n: int = 3,
        num_iterations: int = 100,
        probability: float = 0.99999999,
    ) -> typing.Tuple[core.Tensor, core.Tensor]:
        """
        Segments a plane in the point cloud using the RANSAC algorithm.
        This is a wrapper for a CPU implementation and a copy of the point cloud data and
        resulting plane model and inlier indiecs will be made.

        Args:
            distance_threshold (default 0.01). Max distance a point can be from the plane
            model, and still be considered an inlier.
            ransac_n (default 3). Number of initial points to be considered inliers in each iteration.
            num_iterations (default 100). Maximum number of iterations.
            probability (default 0.99999999). Expected probability of finding the optimal plane.

        Return:
            Tuple of the plane model ax + by + cz + d = 0 and the indices of
            the plane inliers on the same device as the point cloud.

        Example:
            We use Redwood dataset to compute its plane model and inliers::

                sample_pcd_data = o3d.data.PCDPointCloud()
                pcd = o3d.t.io.read_point_cloud(sample_pcd_data.path)
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                         ransac_n=3,
                                                         num_iterations=1000)
                inlier_cloud = pcd.select_by_index(inliers)
                inlier_cloud.paint_uniform_color([1.0, 0, 0])
                outlier_cloud = pcd.select_by_index(inliers, invert=True)
                o3d.visualization.draw([inlier_cloud, outlier_cloud])
        """
    def to(self, device: core.Device, copy: bool = False) -> PointCloud:
        """Transfer the point cloud to a specified device."""

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

class TriangleMesh(Geometry):
    triangle: TensorMap
    vertex: TensorMap
    @overload
    def __init__(self, device: core.Device = core.Device("CPU:0")) -> None: ...
    @overload
    def __init__(self, vertex_positions: core.Tensor, triangle_indices: core.Tensor) -> None: ...
    def clear(self) -> TriangleMesh: ...
    def clone(self) -> TriangleMesh: ...
    def cpu(self) -> TriangleMesh: ...
    def cuda(self, device_id: int = 0) -> TriangleMesh: ...
    @staticmethod
    def from_legacy(
        mesh_legacy: geometry.TriangleMesh,
        vertex_dtype: core.Dtype = core.float32,
        triangle_dtype: core.Dtype = core.int64,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh:
        """
        Create a TriangleMesh from a legacy Open3D TriangleMesh.
        """
    def get_center(self) -> core.Tensor: ...
    def get_max_bound(self) -> core.Tensor: ...
    def get_min_bound(self) -> core.Tensor: ...
    def has_valid_material(self) -> bool: ...
    def get_axis_aligned_bounding_box(self) -> AxisAlignedBoundingBox: ...
    def rotate(self, R: core.Tensor, center: core.Tensor) -> TriangleMesh: ...
    def scale(self, scale: float, center: core.Tensor) -> TriangleMesh: ...
    def to(self, device: core.Device, copy: bool = False) -> TriangleMesh: ...
    def to_legacy(self) -> geometry.TriangleMesh: ...
    def transform(self, transformation: core.Tensor) -> TriangleMesh: ...
    def translate(self, translation: core.Tensor, relative: bool = True) -> TriangleMesh: ...
    def bake_triangle_attr_textures(
        self,
        size: int,
        triangle_attr: typing.Set[str],
        margin: float = 2.0,
        fill: float = 0.0,
        update_material: bool = True,
    ) -> typing.Dict[str, open3d.cpu.pybind.core.Tensor]:
        """
        Bake triangle attributes into textures.

        This function assumes a triangle attribute with name 'texture_uvs'.

        This function always uses the CPU device.

        Args:
            size (int): The width and height of the texture in pixels. Only square
                textures are supported.

            triangle_attr (set): The vertex attributes for which textures should be
                generated.

            margin (float): The margin in pixels. The recommended value is 2. The margin
                are additional pixels around the UV islands to avoid discontinuities.

            fill (float): The value used for filling texels outside the UV islands.

            update_material (bool): If true updates the material of the mesh.
                Baking a vertex attribute with the name 'albedo' will become the albedo
                texture in the material. Existing textures in the material will be
                overwritten.

        Returns:
            A dictionary of tensors that store the baked textures.

        Example:
            We generate a texture visualizing the index of the triangle to which the
            texel belongs to::
                import open3d as o3d
                from matplotlib import pyplot as plt

                box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
                box = o3d.t.geometry.TriangleMesh.from_legacy(box)
                # Creates a triangle attribute 'albedo' which is the triangle index
                # multiplied by (255//12).
                box.triangle['albedo'] = (255//12)*np.arange(box.triangle.indices.shape[0], dtype=np.uint8)

                # Initialize material and bake the 'albedo' triangle attribute to a
                # texture. The texture will be automatically added to the material of
                # the object.
                box.material.set_default_properties()
                texture_tensors = box.bake_triangle_attr_textures(128, {'albedo'})

                # Shows the textured cube.
                o3d.visualization.draw([box])

                # Plot the tensor with the texture.
                plt.imshow(texture_tensors['albedo'].numpy())
        """
    def bake_vertex_attr_textures(
        self,
        size: int,
        vertex_attr: typing.Set[str],
        margin: float = 2.0,
        fill: float = 0.0,
        update_material: bool = True,
    ) -> typing.Dict[str, open3d.cpu.pybind.core.Tensor]:
        """
        Bake vertex attributes into textures.

        This function assumes a triangle attribute with name 'texture_uvs'.
        Only float type attributes can be baked to textures.

        This function always uses the CPU device.

        Args:
            size (int): The width and height of the texture in pixels. Only square
                textures are supported.

            vertex_attr (set): The vertex attributes for which textures should be
                generated.

            margin (float): The margin in pixels. The recommended value is 2. The margin
                are additional pixels around the UV islands to avoid discontinuities.

            fill (float): The value used for filling texels outside the UV islands.

            update_material (bool): If true updates the material of the mesh.
                Baking a vertex attribute with the name 'albedo' will become the albedo
                texture in the material. Existing textures in the material will be
                overwritten.

        Returns:
            A dictionary of tensors that store the baked textures.

        Example:
            We generate a texture storing the xyz coordinates for each texel::
                import open3d as o3d
                from matplotlib import pyplot as plt

                box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
                box = o3d.t.geometry.TriangleMesh.from_legacy(box)
                box.vertex['albedo'] = box.vertex.positions

                # Initialize material and bake the 'albedo' vertex attribute to a
                # texture. The texture will be automatically added to the material of
                # the object.
                box.material.set_default_properties()
                texture_tensors = box.bake_vertex_attr_textures(128, {'albedo'})

                # Shows the textured cube.
                o3d.visualization.draw([box])

                # Plot the tensor with the texture.
                plt.imshow(texture_tensors['albedo'].numpy())
        """
    def boolean_difference(self, mesh: TriangleMesh, tolerance: float = 1e-06) -> TriangleMesh:
        """
        Computes the mesh that encompasses the volume after subtracting the volume of the second operand.
        Both meshes should be manifold.

        This function always uses the CPU device.

        Args:
            mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the
                boolean operation.

            tolerance (float): Threshold which determines when point distances are
                considered to be 0.

        Returns:
            The mesh describing the difference volume.

        Example:
            This subtracts the sphere from the cube volume::

                box = o3d.geometry.TriangleMesh.create_box()
                box = o3d.t.geometry.TriangleMesh.from_legacy(box)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
                sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

                ans = box.boolean_difference(sphere)

                o3d.visualization.draw([{'name': 'difference', 'geometry': ans}])
        """
    def boolean_intersection(self, mesh: TriangleMesh, tolerance: float = 1e-06) -> TriangleMesh:
        """
        Computes the mesh that encompasses the intersection of the volumes of two meshes.
        Both meshes should be manifold.

        This function always uses the CPU device.

        Args:
            mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the
                boolean operation.

            tolerance (float): Threshold which determines when point distances are
                considered to be 0.

        Returns:
            The mesh describing the intersection volume.

        Example:
            This copmutes the intersection of a sphere and a cube::

                box = o3d.geometry.TriangleMesh.create_box()
                box = o3d.t.geometry.TriangleMesh.from_legacy(box)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
                sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

                ans = box.boolean_intersection(sphere)

                o3d.visualization.draw([{'name': 'intersection', 'geometry': ans}])
        """
    def boolean_union(self, mesh: TriangleMesh, tolerance: float = 1e-06) -> TriangleMesh:
        """
        Computes the mesh that encompasses the union of the volumes of two meshes.
        Both meshes should be manifold.

        This function always uses the CPU device.

        Args:
            mesh (open3d.t.geometry.TriangleMesh): This is the second operand for the
                boolean operation.

            tolerance (float): Threshold which determines when point distances are
                considered to be 0.

        Returns:
            The mesh describing the union volume.

        Example:
            This copmutes the union of a sphere and a cube::

                box = o3d.geometry.TriangleMesh.create_box()
                box = o3d.t.geometry.TriangleMesh.from_legacy(box)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.8)
                sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

                ans = box.boolean_union(sphere)

                o3d.visualization.draw([{'name': 'union', 'geometry': ans}])
        """
    def compute_convex_hull(self, joggle_inputs: bool = False) -> Tuple[open3d.t.geometry.TriangleMesh, List[int]]:
        """
        Compute the convex hull of a point cloud using qhull. This runs on the CPU.

        Args:
            joggle_inputs: handle precision problems by randomly perturbing the input data. Set to True if perturbing
              the input is acceptable but you need convex simplicial output. If False, neighboring facets may be merged
              in case of precision problems. See `QHull docs <http://www.qhull.org/html/qh-impre.htm#joggle`__ for more
              details.

        Returns:
            TriangleMesh representing the convexh hull. This contains an extra vertex property "point_indices" that
            contains the index of the corresponding vertex in the original mesh.

        Example::

            # We will load the Stanford Bunny dataset, compute and display it's convex hull
            bunny = o3d.data.BunnyMesh()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
            hull = mesh.compute_convex_hull()
            o3d.visualization.draw([{'name': 'bunny', 'geometry': mesh}, {'name': 'convex hull', 'geometry': hull}])
        """
        ...
    def compute_uvatlas(self, size: int = 512, gutter: float = 1.0, max_stretch: float = 0.1666666716337204) -> None:
        """
        Creates an UV atlas and adds it as triangle attr 'texture_uvs' to the mesh.

        Input meshes must be manifold for this method to work.
        The algorithm is based on:
        Zhou et al, "Iso-charts: Stretch-driven Mesh Parameterization using Spectral
                     Analysis", Eurographics Symposium on Geometry Processing (2004)
        Sander et al. "Signal-Specialized Parametrization" Europgraphics 2002
        This function always uses the CPU device.
        Args:
            size (int): The target size of the texture (size x size). The uv coordinates
                will still be in the range [0..1] but parameters like gutter use pixels
                as units.
            gutter (float): This is the space around the uv islands in pixels.
            max_stretch (float): The maximum amount of stretching allowed. The parameter
                range is [0..1] with 0 meaning no stretch allowed.
        Returns:
            None. This function modifies the mesh in-place.
        Example:
            This code creates a uv map for the Stanford Bunny mesh::
                import open3d as o3d
                bunny = o3d.data.BunnyMesh()
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
                mesh.compute_uvatlas()

                # Add a wood texture and visualize
                texture_data = o3d.data.WoodTexture()
                mesh.material.material_name = 'defaultLit'
                mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(texture_data.albedo_texture_path)
                o3d.visualization.draw(mesh)
        """
    def clip_plane(self, point: ArrayLike, normal: ArrayLike) -> TriangleMesh:
        """
        Returns a new triangle mesh clipped with the plane.

        This method clips the triangle mesh with the specified plane. Parts of the mesh on the positive side of the
        plane will be kept and triangles intersected by the plane will be cut.

        Args:
            point: point on the plane.
            normal: normal of the plane. The normal points to the positive side of the plane for which the geometry
              will be kept.

        Returns:
            New triangle mesh clipped with the plane.

        Example::

            # This example shows how to create a hemisphere from a sphere
            import open3d as o3d
            sphere = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_sphere())
            hemisphere = sphere.clip_plane(point=[0,0,0], normal=[1,0,0])
            o3d.visualization.draw(hemisphere)
        """
    def slice_plane(self, point: ArrayLike, normal: ArrayLike, contour_values: list[float]) -> LineSet:
        """
        Returns a line set with the contour slices defined by the plane and values.

        This method generates slices as LineSet from the mesh at specific contour values with respect to a plane.

        Args:
            point (open3d.core.Tensor): A point on the plane.
            normal (open3d.core.Tensor): The normal of the plane.
            contour_values (list): A list of contour values at which slices will be
                generated. The value describes the signed distance to the plane.

        Returns:
            LineSet with he extracted contours.

        Example::

            # This example shows how to create a hemisphere from a sphere
            import open3d as o3d
            import numpy as np
            bunny = o3d.data.BunnyMesh()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
            contours = mesh.slice_plane([0,0,0], [0,1,0], np.linspace(0,0.2))
            o3d.visualization.draw([{'name': 'bunny', 'geometry': contours}])
        """
    def simplify_quadric_decimation(self, target_reduction: float, preserve_volume: bool = True) -> TriangleMesh:
        """
        Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert.

        This function always uses the CPU device.

        Args:
            target_reduction (float): The factor of triangles to delete, i.e., setting
                this to 0.9 will return a mesh with about 10% of the original triangle
                count. It is not guaranteed that the target reduction factor will be
                reached.

            preserve_volume (bool): If set to True this enables volume preservation
                which reduces the error in triangle normal direction.

        Returns:
            Simplified TriangleMesh.

        Example:
            This shows how to simplifify the Stanford Bunny mesh::

                bunny = o3d.data.BunnyMesh()
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
                simplified = mesh.simplify_quadric_decimation(0.99)
                o3d.visualization.draw([{'name': 'bunny', 'geometry': simplified}])
        """
    @classmethod
    def create_arrow(
        cls,
        cylinder_radius: float = 1.0,
        cone_radius: float = 1.5,
        cylinder_height: float = 5.0,
        cone_height: float = 4.0,
        resolution: int = 20,
        cylinder_split: int = 4,
        cone_split: int = 1,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    @classmethod
    def create_box(
        cls,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    @classmethod
    def create_cone(
        cls,
        radius: float = 1.0,
        height: float = 1.0,
        resolution: int = 20,
        split: int = 4,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    @classmethod
    def create_cylinder(
        cls,
        radius: float = 1.0,
        height: float = 1.0,
        resolution: int = 20,
        split: int = 4,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    @classmethod
    def create_sphere(
        cls,
        radius: float = 1.0,
        resolution: int = 20,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh:
        """
        Factory function to create a sphere mesh centered at (0, 0, 0).

        Args:
            radius (float, optional, default=1.0): The radius of the sphere.
            resolution (int, optional, default=20): The resolution of the sphere. The longitues will be split into resolution segments (i.e. there are resolution + 1 latitude lines including the north and south pole). The latitudes will be split into `2 * resolution segments (i.e. there are 2 * resolution longitude lines.)
            create_uv_map (bool, optional, default=False): Add default uv map to the mesh.

        Returns:
            open3d.geometry.TriangleMesh
        """
    @classmethod
    def create_coordinate_frame(
        cls,
        size: float = 1.0,
        origin: ArrayLike = (0.0, 0.0, 0.0),
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int32,
        device: core.Device = core.Device("CPU:0"),
    ) -> TriangleMesh: ...
    @classmethod
    def create_text(
        text: str,
        depth: float,
        float_dtype: Dtype = core.float32,
        int_dtype: Dtype = core.int64,
        device: core.Device = core.Device("CPU:0"),
    ) -> geometry.TriangleMesh:
        """
        Create a triangle mesh from a text string.

        Args:
            text (str): The text for generating the mesh. ASCII characters 32-126 are
                supported (includes alphanumeric characters and punctuation). In
                addition the line feed '\n' is supported to start a new line.
            depth (float): The depth of the generated mesh. If depth is 0 then a flat mesh will be generated.
            float_dtype (o3d.core.Dtype): Float type for the vertices. Either Float32 or Float64.
            int_dtype (o3d.core.Dtype): Int type for the triangle indices. Either Int32 or Int64.
            device (o3d.core.Device): The device for the returned mesh.

        Returns:
            Text as triangle mesh.

        Example:
            This shows how to simplifify the Stanford Bunny mesh::

                import open3d as o3d

                mesh = o3d.t.geometry.TriangleMesh.create_text('Open3D', depth=1)
                o3d.visualization.draw([{'name': 'text', 'geometry': mesh}])
        """
    def extrude_linear(self, vector: core.Tensor, scale: float = 1.0, capping: bool = True) -> TriangleMesh:
        """
        Sweeps the line set along a direction vector.
        Args:

            vector (open3d.core.Tensor): The direction vector.

            scale (float): Scalar factor which essentially scales the direction vector.
        Returns:
            A triangle mesh with the result of the sweep operation.
        Example:
            This code generates a wedge from a triangle::
                import open3d as o3d
                triangle = o3d.t.geometry.TriangleMesh([[1.0,1.0,0.0], [0,1,0], [1,0,0]], [[0,1,2]])
                wedge = triangle.extrude_linear([0,0,1])
                o3d.visualization.draw([{'name': 'wedge', 'geometry': wedge}])
        """
    def extrude_rotation(
        self,
        angle: float,
        axis: core.Tensor,
        resolution: int = 16,
        translation: float = 0.0,
        capping: bool = True,
    ) -> TriangleMesh:
        """
        Sweeps the triangle mesh rotationally about an axis.
        Args:
            angle (float): The rotation angle in degree.

            axis (open3d.core.Tensor): The rotation axis.

            resolution (int): The resolution defines the number of intermediate sweeps
                about the rotation axis.
            translation (float): The translation along the rotation axis.
        Returns:
            A triangle mesh with the result of the sweep operation.
        Example:
            This code generates a spring with a triangle cross-section::
                import open3d as o3d

                mesh = o3d.t.geometry.TriangleMesh([[1,1,0], [0.7,1,0], [1,0.7,0]], [[0,1,2]])
                spring = mesh.extrude_rotation(3*360, [0,1,0], resolution=3*16, translation=2)
                o3d.visualization.draw([{'name': 'spring', 'geometry': spring}])
        """
    def fill_holes(self, hole_size: float = 1000000.0) -> TriangleMesh:
        """
        Fill holes by triangulating boundary edges.

        This function always uses the CPU device.

        Args:
            hole_size (float): This is the approximate threshold for filling holes.
                The value describes the maximum radius of holes to be filled.

        Returns:
            New mesh after filling holes.

        Example:
            Fill holes at the bottom of the Stanford Bunny mesh::

                bunny = o3d.data.BunnyMesh()
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(bunny.path))
                filled = mesh.fill_holes()
                o3d.visualization.draw([{'name': 'filled', 'geometry': ans}])
        """

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
    def extract_point_cloud(self, weight_threshold: float = 3.0, estimated_point_number: int = -1) -> PointCloud: ...
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
    def voxel_coordinates_and_flattened_indices(self, buf_indices: core.Tensor) -> Tuple[core.Tensor, core.Tensor]: ...
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
    def test_occlusions(
        self, rays: core.Tensor, tnear: float = 0.0, tfar: float = Inf, nthreads: int = 0
    ) -> core.Tensor:
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
    def create_rays_pinhole(
        intrinsic_matrix: core.Tensor, extrinsic_matrix: core.Tensor, width_px: int, height_px: int
    ) -> core.Tensor:
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
    def create_rays_pinhole(
        fov_deg: float,
        center: core.Tensor | ArrayLike,
        eye: core.Tensor | ArrayLike,
        up: core.Tensor | ArrayLike,
        width_px: int,
        height_px: int,
    ) -> core.Tensor:
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
