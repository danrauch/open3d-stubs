from enum import Enum
from typing import Callable, Optional, Tuple, List, Union, overload

from . import utility, camera, pipelines
from numpy import float64, int32, array
from numpy.typing import ArrayLike, NDArray
from numpy import Inf

# pylint: skip-file


class Geometry:
    class GeometryType(Enum):
        Unspecified = 0
        PointCloud = 1
        VoxelGrid = 2
        Octree = 3
        LineSet = 4
        MeshBase = 5
        TriangleMesh = 6
        HalfEdgeTriangleMesh = 7
        Image = 8
        RGBDImage = 9
        TetraMesh = 10
        OrientedBoundingBox = 11
        AxisAlignedBoundingBox = 12

    def __init__(self, *args, **kwargs) -> None: ...
    def clear(self) -> Geometry: ...
    def dimension(self) -> int: ...
    def get_geometry_type(self) -> Geometry.GeometryType: ...
    def is_empty(self) -> bool: ...


class Geometry2D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_max_bound(self) -> NDArray[float64]: ...
    def get_min_bound(self) -> NDArray[float64]: ...


class Geometry3D(Geometry):
    def __init__(self, *args, **kwargs) -> None: ...
    def get_axis_aligned_bounding_box(self) -> AxisAlignedBoundingBox: ...
    def get_center(self) -> NDArray[float64]: ...
    def get_max_bound(self) -> NDArray[float64]: ...
    def get_min_bound(self) -> NDArray[float64]: ...
    def get_oriented_bounding_box(self) -> OrientedBoundingBox: ...

    @classmethod
    def get_rotation_matrix_from_axis_angle(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_quaternion(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_xyz(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_xzy(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_yxz(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_yzx(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_zxy(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    @classmethod
    def get_rotation_matrix_from_zyx(
        cls, rotation: NDArray[float64]
    ) -> NDArray[float64]: ...

    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> Geometry3D: ...
    def scale(self, scale: float, center: NDArray[float64]) -> Geometry3D: ...
    def transform(self, transformation: NDArray[float64]) -> Geometry3D: ...

    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> Geometry3D: ...


class PointCloud(Geometry3D):
    colors: utility.Vector3dVector
    covariances: utility.Vector3dVector
    normals: utility.Vector3dVector
    points: utility.Vector3dVector
    def __init__(self, *args, **kwargs): ...
    def __add__(self, cloud: PointCloud) -> PointCloud: ...
    def __iadd__(self, cloud: PointCloud) -> PointCloud: ...

    def cluster_dbscan(
        self, eps: float, min_points: int, print_progress: bool = False
    ) -> utility.IntVector: ...
    def compute_convex_hull(self) -> Tuple[TriangleMesh, List[int]]: ...
    def compute_mahalanobis_distance(self) -> utility.DoubleVector: ...

    def compute_mean_and_covariance(
        self,
    ) -> Tuple[NDArray[float64], NDArray[float64]]: ...
    def compute_nearest_neighbor_distance(self) -> utility.DoubleVector: ...

    def compute_point_cloud_distance(
        self, target: PointCloud
    ) -> utility.DoubleVector: ...

    @classmethod
    def create_from_depth_image(
        cls,
        depth: Image,
        intrinsic,
        extrinsic: NDArray[float64],
        depth_scale: float = 1000.0,
        depth_trunc: float = 1000.0,
        stride: int = 1,
        project_valid_depth_only: bool = True,
    ) -> PointCloud: ...

    @classmethod
    def create_from_rgbd_image(
        cls,
        iamge: Image,
        intrinsic,
        extrinsic: NDArray[float64],
        project_valid_depth_only: bool = True,
    ) -> PointCloud: ...

    def crop(
        self,
        bounding_box: Union[AxisAlignedBoundingBox, OrientedBoundingBox],
    ) -> PointCloud: ...

    def estimate_normals(
        self,
        search_param: KDTreeSearchParam = KDTreeSearchParamKNN(),
        fast_normal_computation: bool = True,
    ) -> None: ...
    def has_colors(self) -> bool: ...
    def has_normals(self) -> bool: ...
    def has_points(self) -> bool: ...

    def hidden_point_removal(
        self, camera_location: ArrayLike, radius: float
    ) -> Tuple[TriangleMesh, List[int]]: ...
    def normalize_normals(self) -> PointCloud: ...
    def orient_normals_consistent_tangent_plane(self, k: int) -> None: ...

    def orient_normals_to_align_with_direction(
        self, orientation_reference: NDArray[float64] = array([0.0, 0.0, 1.0])
    ) -> None: ...

    def orient_normals_towards_camera_location(
        self, camera_location: NDArray[float64] = array([0.0, 0.0, 0.0])
    ) -> None: ...
    def paint_uniform_color(self, color: ArrayLike) -> PointCloud: ...
    def random_down_sample(self, sampling_ratio: float) -> PointCloud: ...

    def remove_non_finite_points(
        self, remove_nan: bool = True, remove_infinite: bool = True
    ) -> PointCloud: ...

    def remove_radius_outlier(
        self, nb_points: int, radius: float
    ) -> Tuple[PointCloud, List[int]]: ...

    def remove_statistical_outlier(
        self, nb_neighbors: int, std_ratio: float
    ) -> Tuple[PointCloud, List[int]]: ...

    def segment_plane(
        self, distance_threshold: float, ransac_n: int, num_iterations: int
    ) -> Tuple[NDArray[float64], List[int]]: ...

    def select_by_index(
        self, indices: List[int], invert: bool = False
    ) -> PointCloud: ...
    def uniform_down_sample(self, every_k_points: int) -> PointCloud: ...
    def voxel_down_sample(self, voxel_size: float) -> PointCloud: ...

    def voxel_down_sample_and_trace(
        self,
        voxel_size: float,
        min_bound: NDArray[float64],
        max_bound: NDArray[float64],
        approximate_class: bool = False,
    ) -> Tuple[PointCloud, NDArray[int32], List[utility.IntVector]]: ...

    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> PointCloud: ...
    def scale(self, scale: float, center: NDArray[float64]) -> PointCloud: ...
    def transform(self, transformation: NDArray[float64]) -> PointCloud: ...

    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> PointCloud: ...


class Image(Geometry2D):
    def __init__(self, *args, **kwargs) -> None: ...

    def create_pyramid(
        self, num_of_levels: int, with_gaussian_filter: bool
    ) -> List[Image]: ...
    def filter(self, filter_type: ImageFilterType) -> Image: ...

    @classmethod
    def filter_pyramid(
        cls, image_pyramid: List[Image], filter_type: ImageFilterType
    ) -> List[Image]: ...
    def flip_horizontal(self) -> Image: ...
    def flip_vertical(self) -> Image: ...


class ImageFilterType(Enum):
    Gaussian5: ...
    Gaussian7: ...
    Sobel3dx: ...
    Sobel3dy: ...


class KDTreeFlann:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, data: NDArray[float64]) -> None: ...
    @overload
    def __init__(self, geometry: Geometry) -> None: ...
    @overload
    def __init__(self, feature: pipelines.registration.Feature) -> None: ...

    def search_hybrid_vector_3d(
        self, query: ArrayLike, radius: float, max_nn: int
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_hybrid_vector_xd(
        self, query: ArrayLike, radius: float, max_nn: int
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_knn_vector_3d(
        self, query: ArrayLike, max_nn: int
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_knn_vector_xd(
        self, query: ArrayLike, max_nn: int
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_radius_vector_3d(
        self, query: ArrayLike, radius: float
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_radius_vector_xd(
        self, query: ArrayLike, radius: float
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_vector_3d(
        self, query: ArrayLike, search_param: KDTreeSearchParam
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...

    def search_vector_xd(
        self, query: ArrayLike, search_param: KDTreeSearchParam
    ) -> Tuple[int, utility.IntVector, utility.DoubleVector]: ...
    def set_feature(self, feature: pipelines.registration.Feature) -> bool: ...
    def set_geometry(self, geometry: Geometry) -> bool: ...
    def set_matrix_data(self, data: NDArray[float64]) -> bool: ...


class KDTreeSearchParam:
    class SearchType(Enum):
        HybridSearch: ...
        KNNSearch: ...
        RadiusSearch: ...

    def __init__(self, *args, **kwargs) -> None: ...
    def get_search_type(self) -> KDTreeSearchParam.SearchType: ...


class KDTreeSearchParamHybrid(KDTreeSearchParam):
    max_nn: int
    radius: float
    def __init__(self, radius: float, max_nn: int) -> None: ...


class KDTreeSearchParamKNN(KDTreeSearchParam):
    knn: int
    def __init__(self, knn: int = 30) -> None: ...


class KDTreeSearchParamRadius(KDTreeSearchParam):
    radius: float
    def __init__(self, radius: float) -> None: ...


class AxisAlignedBoundingBox(Geometry3D):
    color: NDArray[float64] | ArrayLike
    max_bound: NDArray[float64] | ArrayLike
    min_bound: NDArray[float64] | ArrayLike
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, aabb: AxisAlignedBoundingBox) -> None: ...

    @overload
    def __init__(
        self, min_bound: NDArray[float64] | ArrayLike, max_bound: NDArray[float64] | ArrayLike
    ) -> None: ...

    @classmethod
    def create_from_points(
        cls, points: utility.Vector3dVector
    ) -> AxisAlignedBoundingBox: ...
    def get_box_points(self) -> utility.Vector3dVector: ...
    def get_extent(self) -> NDArray[float64]: ...
    def get_half_extent(self) -> NDArray[float64]: ...
    def get_max_extent(self) -> float: ...

    def get_point_indices_within_bounding_box(
        self, points: utility.Vector3dVector
    ) -> List[int]: ...
    def get_print_info(self) -> str: ...
    def volume(self) -> float: ...


class OrientedBoundingBox(Geometry3D):
    color: ArrayLike
    max_bound: ArrayLike
    min_bound: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...

    @classmethod
    def create_from_axis_aligned_bounding_box(
        cls, aabox: AxisAlignedBoundingBox
    ) -> OrientedBoundingBox: ...

    @classmethod
    def create_from_points(
        cls, points: utility.Vector3dVector
    ) -> OrientedBoundingBox: ...
    def get_box_points(self) -> utility.Vector3dVector: ...

    def get_point_indices_within_bounding_box(
        self, points: utility.Vector3dVector
    ) -> List[int]: ...
    def volume(self) -> float: ...


class LineSet(Geometry3D):
    colors: ArrayLike
    lines: ArrayLike
    points: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...

    @overload
    @classmethod
    def create_camera_visualization(
        cls,
        view_width_px: int,
        view_height_px: int,
        intrinsic: NDArray[float64],
        extrinsic: NDArray[float64],
        scale: float = 1.0,
    ) -> LineSet: ...

    @overload
    @classmethod
    def create_camera_visualization(
        cls,
        intrinsic: camera.PinholeCameraIntrinsic,
        extrinsic: NDArray[float64],
        scale: float = 1.0,
    ) -> LineSet: ...

    @classmethod
    def create_from_axis_aligned_bounding_box(
        cls, box: AxisAlignedBoundingBox
    ) -> LineSet: ...
    @classmethod
    def create_from_oriented_bounding_box(cls, box: OrientedBoundingBox) -> LineSet: ...

    @classmethod
    def create_from_point_cloud_correspondences(
        cls,
        cloud0: PointCloud,
        cloud1: PointCloud,
        correspondences: List[Tuple[int, int]],
    ) -> LineSet: ...
    @classmethod
    def create_from_tetra_mesh(cls, mesh: TetraMesh) -> LineSet: ...
    @classmethod
    def create_from_triangle_mesh(cls, mesh: TriangleMesh) -> LineSet: ...
    def paint_uniform_color(self, color: ArrayLike) -> LineSet: ...
    def has_colors(self) -> bool: ...
    def has_lines(self) -> bool: ...
    def has_points(self) -> bool: ...


class MeshBase(Geometry3D):
    vertex_colors: utility.Vector3dVector
    vertex_normals: utility.Vector3dVector
    vertices: utility.Vector3dVector
    def __init__(self, *args, **kwargs) -> None: ...
    def compute_convex_hull(self) -> Tuple[TriangleMesh, List[int]]: ...
    def has_vertex_colors(self) -> bool: ...
    def has_vertex_normals(self) -> bool: ...
    def has_verteices(self) -> bool: ...
    def normalize_normals(self) -> MeshBase: ...


class TriangleMesh(MeshBase):
    adjacency_list: List[set]
    textures: Image
    triangle_material_ids: utility.IntVector
    triangle_normals: utility.Vector3dVector
    triangle_uvs: utility.Vector2dVector
    triangles: utility.Vector3iVector
    def __init__(self, *args, **kwargs) -> None: ...

    def cluster_connected_triangles(
        self,
    ) -> Tuple[utility.IntVector, List[int], utility.DoubleVector]: ...
    def compute_adjacency_list(self) -> TriangleMesh: ...
    def compute_triangle_normals(self, normalized: bool = True) -> TriangleMesh: ...
    def compute_vertex_normals(self, normalized: bool = True) -> TriangleMesh: ...

    def remove_triangles_by_index(self, triangle_indices: List[int]) -> None:
        """
        This function removes the triangles with index in triangle_indices. Call remove_unreferenced_vertices to clean
        up vertices afterwards.

        Args:
            triangle_indices (list[int]): 1D array of triangle indices that should be removed from the TriangleMesh

        Returns:
            None
        """
        ...

    def remove_triangles_by_mask(self, triangle_mask: List[int]) -> None:
        """
        This function removes the triangles where triangle_mask is set to true. Call remove_unreferenced_vertices to 
        clean up vertices afterwards.

        Args:
            triangle_indices (list[int]): 1D bool array, True values indicate triangles that should be removed.

        Returns:
            None
        """
        ...

    def remove_unreferenced_vertices(self) -> TriangleMesh:
        """
        This function removes vertices from the triangle mesh that are not referenced in any triangle of the mesh.

        Returns:
            TriangleMesh
        """
        ...

    def simplify_quadric_decimation(self, target_number_of_triangles: int, maximum_error: float = Inf,
                                    boundary_weight: float = 1.0) -> TriangleMesh:
        """
        Function to simplify mesh using Quadric Error Metric Decimation by Garland and Heckbert.

        Args:
            target_number_of_triangles (int): The number of triangles that the simplified mesh should have. It is not guaranteed that this number will be reached.
            maximum_error (float, optional, default=inf): The maximum error where a vertex is allowed to be merged
            boundary_weight (float, optional, default=1.0): A weight applied to edge vertices used to preserve boundaries

        Returns:
            TriangleMesh
        """
        ...

    def simplify_vertex_clustering(self,
                                   voxel_size: float,
                                   contradiction: SimplificationContraction = SimplificationContraction.Average
                                   ) -> TriangleMesh:
        """
        Function to simplify mesh using vertex clustering.

        Args:
            voxel_size (float): The size of the voxel within vertices are pooled.
            contradiction: Method to aggregate vertex information. Average computes a simple average, Quadric minimizes the distance to the adjacent planes.

        Returns:
            TriangleMesh
        """
        ...

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
    ) -> TriangleMesh: ...

    @classmethod
    def create_box(
        cls,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        create_uv_map: bool = False,
        map_texturea_to_each_face: bool = False,
    ) -> TriangleMesh: ...

    @classmethod
    def create_cone(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        resolution: int = 20,
        split: int = 1,
        create_uv_map: bool = False,
    ) -> TriangleMesh: ...

    @classmethod
    def create_sphere(
        cls,
        radius: float = 1.0,
        resolution: int = 20,
        create_uv_map: bool = False,
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
        ...

    @classmethod
    def create_coordinate_frame(
        cls, size: float = 1.0, origin: ArrayLike = array([0.0, 0.0, 0.0])
    ) -> TriangleMesh: ...

    @classmethod
    def create_from_point_cloud_poisson(
        cls, pcd: PointCloud, depth: int = 8, width: float = 0.0, scale: float = 1.1,
        linear_fit: bool = False, n_threads: int = -1
    ) -> Tuple[TriangleMesh, utility.DoubleVector]:
        """
        Function that computes a triangle mesh from a oriented PointCloud pcd. This implements the Screened Poisson
        Reconstruction proposed in Kazhdan and Hoppe, “Screened Poisson Surface Reconstruction”, 2013. This function
        uses the original implementation by Kazhdan. See https://github.com/mkazhdan/PoissonRecon   

        Args:
            pcd (open3d.geometry.PointCloud): PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
            depth (int, optional, default=8): Maximum depth of the tree that will be used for surface reconstruction. Running at depth d corresponds to solving on a grid whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified reconstruction depth is only an upper bound.
            width (float, optional, default=0): Specifies the target width of the finest level octree cells. This parameter is ignored if depth is specified
            scale (float, optional, default=1.1): Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the samples’ bounding cube.
            linear_fit (bool, optional, default=False): If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
            n_threads (int, optional, default=-1): Number of threads used for reconstruction. Set to -1 to automatically determine it.

        Returns:
            Tuple[open3d.geometry.TriangleMesh, open3d.utility.DoubleVector]
        """
        ...

    @classmethod
    def create_from_point_cloud_ball_pivoting(
        cls, pcd: PointCloud, radii: utility.DoubleVector
    ) -> TriangleMesh:
        """
        Function that computes a triangle mesh from a oriented PointCloud. This implements the Ball Pivoting algorithm
        proposed in F. Bernardini et al., “The ball-pivoting algorithm for surface reconstruction”, 1999. The
        implementation is also based on the algorithms outlined in Digne, “An Analysis and Implementation of a Parallel
        Ball Pivoting Algorithm”, 2014. The surface reconstruction is done by rolling a ball with a given radius over
        the point cloud, whenever the ball touches three points a triangle is created.

        Args:
            pcd (open3d.geometry.PointCloud): PointCloud from which the TriangleMesh surface is reconstructed. Has to contain normals.
            radii (open3d.utility.DoubleVector): The radii of the ball that are used for the surface reconstruction.

        Returns:
            open3d.geometry.TriangleMesh
        """
        ...

    @classmethod
    def create_cylinder(
        cls,
        radius: float = 1.0,
        height: float = 2.0,
        resolution: int = 20,
        split: int = 4,
        create_uv_map: bool = False,
    ) -> TriangleMesh: ...
    def has_adjacency_list(self) -> bool: ...
    def has_textures(self) -> bool: ...
    def has_triangle_material_ids(self) -> bool: ...
    def has_triangle_normals(self) -> bool: ...
    def has_triangle_uvs(self) -> bool: ...
    def has_triangles(self) -> bool: ...
    def has_vertex_colors(self) -> bool: ...
    def has_vertex_normals(self) -> bool: ...
    def has_vertices(self) -> bool: ...
    def is_edge_manifold(self, allow_boundary_edges: bool = True) -> bool: ...
    def is_empty(self) -> bool: ...
    def is_intersecting(self, other: TriangleMesh) -> bool: ...
    def is_orientable(self) -> bool: ...
    def is_self_intersecting(self) -> bool: ...
    def is_vertex_manifold(self) -> bool: ...
    def is_watertight(self) -> bool: ...
    def paint_uniform_color(self, arg0: ArrayLike) -> TriangleMesh: ...

    def sample_points_poisson_disk(self, number_of_points: int,
                                   init_factor: float = 5.0,
                                   pcl: Optional[PointCloud] = None,
                                   use_triangle_normal: bool = False,
                                   seed: int = -1) -> PointCloud:
        """
        Function to sample points from the mesh, where each point has approximately the same distance to the
        neighbouring points (blue noise). Method is based on Yuksel, “Sample Elimination for Generating Poisson Disk
        Sample Sets”, EUROGRAPHICS, 2015.

        Args:
            number_of_points (int): Number of points that should be sampled.
            init_factor (float, optional): Factor for the initial uniformly sampled PointCloud. This init PointCloud is used for sample elimination. Default: 5.
            pcl (open3d.geometry.PointCloud, optional): Initial PointCloud that is used for sample elimination. If this parameter is provided the init_factor is ignored. Default: None.
            use_triangle_normal (bool, optional): If True assigns the triangle normals instead of the interpolated vertex normals to the returned points. The triangle normals will be computed and added to the mesh if necessary. Default: False.
            seed (int, optional): Seed value used in the random generator, set to -1 to use a random seed value with each function call. Default: -1.

        Returns:
            open3d.geometry.PointCloud
        """
        ...


class TetraMesh(MeshBase):
    tetras: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def create_from_point_cloud(cls, point_cloud: PointCloud) -> TetraMesh: ...

    def extract_triangle_mesh(
        self, values: utility.DoubleVector, level: float
    ) -> TriangleMesh: ...
    def has_tetras(self) -> bool: ...
    def remove_degenerate_tetras(self) -> TetraMesh: ...
    def remove_duplicated_tetras(self) -> TetraMesh: ...
    def remove_duplicated_vertices(self) -> TetraMesh: ...
    def remove_unreferenced_vertices(self) -> TetraMesh: ...


class Voxel:
    color: ArrayLike
    grid_index: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...


class VoxelGrid(Geometry3D):
    origin: Tuple[float, float, float]
    voxel_size: float
    def __init__(self, *args, **kwargs) -> None: ...
    def __add__(self, voxelgrid: VoxelGrid) -> VoxelGrid: ...
    def __iadd__(self, voxelgrid: VoxelGrid) -> VoxelGrid: ...

    def carve_depth_map(
        self,
        depth_map: Image,
        camera_params: camera.PinholeCameraParameters,
        keep_voxles_outside_image: bool = False,
    ) -> VoxelGrid: ...

    def carve_silhouette(
        self,
        silhouette_mask: Image,
        camera_params: camera.PinholeCameraParameters,
        keep_voxles_outside_image: bool = False,
    ) -> VoxelGrid: ...
    def check_if_included(self, queries: utility.Vector3dVector) -> List[bool]: ...

    @classmethod
    def create_dense(
        cls,
        origin: ArrayLike,
        color: ArrayLike,
        voxel_size: float,
        width: float,
        height: float,
        depth: float,
    ) -> VoxelGrid: ...
    def create_from_octree(self, octree: Octree) -> None: ...

    @classmethod
    def create_from_point_cloud(
        cls, input: PointCloud, voxel_size: float
    ) -> VoxelGrid: ...

    @classmethod
    def create_from_point_cloud_within_bounds(
        cls,
        input: PointCloud,
        voxel_size: float,
        min_bound: ArrayLike,
        max_bound: ArrayLike,
    ) -> VoxelGrid: ...

    @classmethod
    def create_from_triangle_mesh(
        cls, input: TriangleMesh, voxel_size: float
    ) -> VoxelGrid: ...

    @classmethod
    def create_from_triangle_mesh_within_bounds(
        cls,
        input: TriangleMesh,
        voxel_size: float,
        min_bound: ArrayLike,
        max_bound: ArrayLike,
    ) -> VoxelGrid: ...
    def get_voxel(self, point: NDArray[float64]) -> NDArray[float64]: ...
    def get_voxels(self) -> List[Voxel]: ...
    def has_colors(self) -> bool: ...
    def has_voxels(self) -> bool: ...
    def to_octree(self, max_depth: int) -> Octree: ...

    def rotate(
        self, R: NDArray[float64], center: NDArray[float64] = ...
    ) -> VoxelGrid: ...
    def scale(self, scale: float, center: NDArray[float64]) -> VoxelGrid: ...
    def transform(self, transformation: NDArray[float64]) -> VoxelGrid: ...

    def translate(
        self, translation: NDArray[float64], relative: bool = True
    ) -> VoxelGrid: ...


class RGBDImage(Geometry2D):
    color: Image
    depth: Image
    def __init__(self) -> None: ...

    @classmethod
    def create_from_color_and_depth(
        cls,
        color: Image,
        depth: Image,
        depth_scale: float = 1000.0,
        depth_trunc: float = 3.0,
        convert_rgb_to_intensity: bool = True,
    ) -> RGBDImage: ...

    @classmethod
    def create_from_nyu_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...

    @classmethod
    def create_from_redwood_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...

    @classmethod
    def create_from_sun_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...

    @classmethod
    def create_from_tum_format(
        cls, color: Image, depth: Image, convert_rgb_to_intensity: bool = True
    ) -> RGBDImage: ...


class Octree(Geometry3D):
    max_depth: int
    origin: NDArray[float64]
    root_node: OctreeNode
    size: float
    def __init__(self, *args, **kwargs) -> None: ...

    def convert_from_point_cloud(
        self, point_cloud: PointCloud, size_expand: float = 0.01
    ) -> None: ...
    def create_from_voxel_grid(self, voxel_grid: VoxelGrid) -> None: ...

    def insert_point(
        self,
        point: NDArray[float64],
        f_init: Callable[[], OctreeLeafNode],
        f_update: Callable[[OctreeLeafNode], None],
        fi_init: Optional[Callable[[], OctreeInternalNode]] = None,
        fi_update: Optional[Callable[[OctreeInternalNode], None]] = None,
    ) -> None: ...

    @classmethod
    def is_point_in_bound(
        cls, point: NDArray[float64], origin: NDArray[float64], size: float
    ) -> bool: ...

    def locate_leaf_node(
        self, point: NDArray[float64]
    ) -> Tuple[OctreeLeafNode, OctreeNodeInfo]: ...
    def to_voxel_grid(self) -> VoxelGrid: ...
    def traverse(self, f: Callable[[OctreeNode, OctreeNodeInfo], bool]) -> None: ...


class OctreeNode:
    def __init__(self, *args, **kwargs) -> None: ...


class OctreeNodeInfo:
    child_index: int
    depth: int
    origin: NDArray[float64]
    size: float
    def __init__(self, *args, **kwargs) -> None: ...


class OctreeLeafNode(OctreeNode):
    def __init__(self, *args, **kwargs) -> None: ...
    def clone(self) -> OctreeLeafNode: ...


class OctreeColorLeafNode(OctreeLeafNode):
    color: NDArray[float64]
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_init_function(cls) -> Callable[[], OctreeLeafNode]: ...

    @classmethod
    def get_update_function(
        cls, color: NDArray[float64]
    ) -> Callable[[OctreeLeafNode], None]: ...


class OctreePointColorLeafNode(OctreeColorLeafNode):
    indices: List[int]
    def __init__(self, *args, **kwargs) -> None: ...


class OctreeInternalNode(OctreeNode):
    children: List[OctreeNode]
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_init_function(cls) -> Callable[[], OctreeInternalNode]: ...
    @classmethod
    def get_update_function(cls) -> Callable[[OctreeInternalNode], None]: ...


class OctreeInternalPointNode(OctreeInternalNode):
    indices: List[int]


class DeformAsRigidAsPossibleEnergy(Enum):
    Smoothed = ...
    Spokes = ...


class FilterScope(Enum):
    All = ...
    Color = ...
    Normal = ...
    Vertex = ...


class SimplificationContraction(Enum):
    Average = ...
    Quadric = ...


class HalfEdge:
    next: int
    triangle_index: int
    twin: int
    vertex_indices: List[int]
    def __init__(self, *args, **kwargs) -> None: ...
    def is_boundary(self) -> bool: ...


class HalfEdgeTriangleMesh(MeshBase):
    half_edges: List[HalfEdge]
    ordered_half_edge_from_vertex: List[List[int]]
    triangle_normals: utility.Vector3dVector
    triangles: utility.Vector3iVector
    def __init__(self, *args, **kwargs) -> None: ...

    def boundary_half_edges_from_vertex(
        self, vertex_index: int
    ) -> utility.IntVector: ...
    def boundary_vertices_from_vertex(self, vertex_index: int) -> utility.IntVector: ...
    @classmethod
    def create_from_triangle_mesh(cls, mesh: TriangleMesh) -> HalfEdgeTriangleMesh: ...
    def has_half_edges(self) -> bool: ...
