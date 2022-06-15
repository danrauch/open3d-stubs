from typing import overload, List, Union, Iterable, Optional
from numpy import float64, int32
from numpy.typing import NDArray
from .. import geometry, utility


class Feature:
    data: NDArray[float64]
    def __init__(self, *args, **kwargs) -> None: ...
    def dimension(self) -> int: ...
    def num(self) -> int: ...
    def resize(self, dim: int, n: int) -> None: ...


class RobustKernel:
    def __init__(self, *args, **kwargs) -> None: ...
    def weight(self, residual: float) -> float: ...


class CauchyLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...


class GMLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...


class HuberLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...


class L1Loss(RobustKernel):
    def __init__(self, *args, **kwargs) -> None: ...


class L2Loss(RobustKernel):
    def __init__(self, *args, **kwargs) -> None: ...


class TukeyLoss(RobustKernel):
    k: float
    def __init__(self, *args, **kwargs) -> None: ...


class CorrespondenceChecker:
    require_pointcloud_alighment_: bool
    def __init__(self, *args, **kwargs) -> None: ...

    def Check(
        self,
        source: geometry.PointCloud,
        target: geometry.PointCloud,
        corres: utility.Vector2iVector,
        transformation: NDArray[float64],
    ) -> bool: ...


class CorrespondenceCheckerBasedOnDistance(CorrespondenceChecker):
    distance_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...


class CorrespondenceCheckerBasedOnEdgeLength(CorrespondenceChecker):
    similarity_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...


class CorrespondenceCheckerBasedOnNormal(CorrespondenceChecker):
    normal_angle_threshold: float
    def __init__(self, *args, **kwargs) -> None: ...


class ICPConvergenceCriteria:
    max_iteration: int
    relative_fitness: float
    relative_rmse: float
    @overload
    def __init__(self, other: ICPConvergenceCriteria) -> None: ...

    @overload
    def __init__(
        self,
        relative_fitness: float = 1e-06,
        relative_rmse: float = 1e-06,
        max_iteration: int = 30,
    ) -> None: ...


class PoseGraphEdge:
    confidence: float
    information: NDArray[float64]
    source_node_id: int
    target_node_id: int
    transformation: NDArray[float64]
    uncertain: bool
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: PoseGraphEdge) -> None: ...

    @overload
    def __init__(
        self,
        source_node_id: int = -1,
        target_node_id: int = -1,
        transformation: NDArray[float64] = ...,
        information: NDArray[float64] = ...,
        uncertain: bool = False,
        confidence: float = 1.0,
    ) -> None: ...


class PoseGraphEdgeVector:
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> PoseGraphEdge: ...
    def __setitem__(self, key, value: PoseGraphEdge) -> PoseGraphEdgeVector: ...
    def append(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def extend(self, L: Union[PoseGraphEdgeVector, Iterable]) -> None: ...
    def insert(self, i: int, x: PoseGraphEdge) -> None: ...
    def pop(self, i: Optional[int]) -> PoseGraphEdge: ...


class PoseGraphNode:
    pose: NDArray[float64]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: PoseGraphNode) -> None: ...

    @overload
    def __init__(
        self,
        pose: NDArray[float64],
    ) -> None: ...


class PoseGraphNodeVector:
    def __init__(self, *args, **kwargs) -> None: ...
    def __getitem__(self, key) -> PoseGraphNode: ...
    def __setitem__(self, key, value: PoseGraphNode) -> PoseGraphNodeVector: ...
    def append(self, x: float) -> None: ...
    def clear(self) -> None: ...
    def extend(self, L: Union[PoseGraphNodeVector, Iterable]) -> None: ...
    def insert(self, i: int, x: PoseGraphNode) -> None: ...
    def pop(self, i: Optional[int]) -> PoseGraphNode: ...


class PoseGraph:
    edges: List[PoseGraphEdge]
    nodes: List[PoseGraphNode]
    def __init__(self, *args, **kwargs) -> None: ...


class FastGlobalRegistrationOption:
    decrease_mu: bool
    division_factor: float
    iteration_number: int
    maximum_correspondence_distance: float
    maximum_tuple_count: float
    seed: int
    tuple_scale: float
    tuple_test: bool
    use_absolute_scale: bool
    @overload
    def __init__(self, other: FastGlobalRegistrationOption) -> None: ...

    @overload
    def __init__(
        self,
        division_factor: float = 1.4,
        use_absolute_scale: bool = False,
        decrease_mu: bool = False,
        maximum_correspondence_distance: float = 0.025,
        iteration_number: int = 64,
        tuple_scale: float = 0.95,
        tuple_test: bool = True,
        maximum_tuple_count: int = 1000,
        seed: Optional[int] = None
    ) -> None: ...


class RANSACConvergenceCriteria:
    confidence: float
    max_iteration: int
    @overload
    def __init__(self, other: RANSACConvergenceCriteria) -> None: ...

    @overload
    def __init__(
        self, max_iteration: int = 100000, confidence: float = 0.999
    ) -> None: ...


class RegistrationResult:
    correspondence_set: NDArray[int32]
    fitness: float
    inlier_rmse: float
    transformation: NDArray[float64]
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: RegistrationResult) -> None: ...


class GlobalOptimizationConvergenceCriteria:
    lower_scale_factor: float
    max_iteration: int
    max_iteration_lm: int
    min_relative_increment: float
    min_relative_residual_increment: float
    min_residual: float
    min_right_term: float
    upper_scale_factor: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: GlobalOptimizationConvergenceCriteria) -> None: ...


class GlobalOptimizationOption:
    edge_prune_threshold: float
    max_correspondence_distance: float
    preference_loop_closure: float
    reference_node: int
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other: GlobalOptimizationOption) -> None: ...

    @overload
    def __init__(self, max_correspondence_distance: float = 0.03, edge_prune_threshold: float = 0.25,
                 preference_loop_closure: float = 1.0, reference_node: int = -1) -> None:
        ...


class GlobalOptimizationMethod:
    """Base class for global optimization method."""

    def __init__(self) -> None: ...

    def optimize_pose_graph(self,
                            pose_graph: PoseGraph,
                            criteria: GlobalOptimizationConvergenceCriteria,
                            option: GlobalOptimizationOption) -> None:
        """Run pose graph optimization."""
        ...


class GlobalOptimizationLevenbergMarquardt(GlobalOptimizationMethod):
    """
    Global optimization with Levenberg-Marquardt algorithm. Recommended over the Gauss-Newton method since the LM has
    better convergence characteristics.
    """
    ...


class GlobalOptimizationGaussNewton(GlobalOptimizationMethod):
    """Global optimization with Gauss-Newton algorithm."""
    ...


class TransformationEstimation:
    """
    Base class that estimates a transformation between two point clouds. The virtual function ComputeTransformation()
    must be implemented in subclasses.
    """

    def __init__(self, *args, **kwargs) -> None: ...

    def compute_rmse(self, source: geometry.PointCloud, target: geometry.PointCloud,
                     corres: utility.Vector2iVector) -> float:
        """Compute RMSE between source and target points cloud given correspondences."""
        ...

    def compute_transformation(self, source: geometry.PointCloud, target: geometry.PointCloud,
                               corres: utility.Vector2iVector) -> float:
        """Compute transformation from source to target point cloud given correspondences."""
        ...


class TransformationEstimationPointToPoint(TransformationEstimation):
    with_scaling: RobustKernel
    ...


class TransformationEstimationPointToPlane(TransformationEstimation):
    kernel: bool
    ...


def compute_fpfh_feature(
    input: geometry.PointCloud,
    search_param: geometry.KDTreeSearchParam
) -> Feature:
    """
    Function to compute FPFH feature for a point cloud

    Args:
        input (open3d.geometry.PointCloud): The Input point cloud.
        search_param (open3d.geometry.KDTreeSearchParam): KDTree KNN search parameter.

    Returns:
        open3d.pipelines.registration.Feature
    """
    ...


def evaluate_registration(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    transformation: NDArray[float64]
) -> RegistrationResult:
    """
    Function for evaluating registration between point clouds.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (numpy.ndarray[numpy.float64[4, 4]], optional):
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...


def get_information_matrix_from_point_clouds(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    transformation: NDArray[float64] = None
) -> NDArray[float64]:
    """
    Function for computing information matrix from transformation matrix.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (numpy.ndarray[numpy.float64[4, 4]], optional):
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    Returns:
        numpy.ndarray[numpy.float64[6, 6]]
    """
    ...


def global_optimization(
    pose_graph: PoseGraph,
    method: GlobalOptimizationMethod,
    max_correspondence_distance: float,
    transformation: NDArray[float64] = None
) -> NDArray[float64]:
    """
    Function for computing information matrix from transformation matrix.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        transformation (numpy.ndarray[numpy.float64[4, 4]], optional):
            The 4x4 transformation matrix to transform source to target Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    Returns:
        numpy.ndarray[numpy.float64[6, 6]]
    """
    ...


def registration_icp(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    max_correspondence_distance: float,
    init: NDArray[float64] = None,
    estimation_method: TransformationEstimation = TransformationEstimationPointToPoint(),
    criteria: ICPConvergenceCriteria = ICPConvergenceCriteria()
) -> RegistrationResult:
    """
    Function for ICP registration.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        init (numpy.ndarray[numpy.float64[4, 4]], optional):
            Initial transformation estimation Default value:
            array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        estimation_method (open3d.pipelines.registration.TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method. One of (TransformationEstimationPointToPoint, TransformationEstimationPointToPlane, TransformationEstimationForGeneralizedICP, TransformationEstimationForColoredICP)
        criteria (open3d.pipelines.registration.ICPConvergenceCriteria, optional, default=ICPConvergenceCriteria class with relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, and max_iteration=30): Convergence criteria

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...


def registration_fgr_based_on_feature_matching(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    source_feature: Feature,
    target_feature: Feature,
    option: FastGlobalRegistrationOption = FastGlobalRegistrationOption()
) -> RegistrationResult:
    """
    Function for global RANSAC registration based on feature matching.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        source_feature (open3d.pipelines.registration.Feature): Source point cloud feature.
        target_feature (open3d.pipelines.registration.Feature): Target point cloud feature.
        option (open3d.pipelines.registration.FastGlobalRegistrationOption, optional):
            Registration option Default value: FastGlobalRegistrationOption class with division_factor= tuple_test={} 
            use_absolute_scale= seed={} decrease_mu=1.4 maximum_correspondence_distance=false iteration_number=true
            tuple_scale=0.025 maximum_tuple_count=64

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...


def registration_ransac_based_on_feature_matching(
    source: geometry.PointCloud,
    target: geometry.PointCloud,
    source_feature: Feature,
    target_feature: Feature,
    mutual_filter: bool,
    max_correspondence_distance: float,
    estimation_method: TransformationEstimation = TransformationEstimationPointToPoint(),
    ransac_n: int = 3,
    checkers: List[CorrespondenceChecker] = [],
    criteria: ICPConvergenceCriteria = ICPConvergenceCriteria(),
    seed: Optional[int] = None
) -> RegistrationResult:
    """
    Function for global RANSAC registration based on feature matching.

    Args:
        source (open3d.geometry.PointCloud): The source point cloud.
        target (open3d.geometry.PointCloud): The target point cloud.
        source_feature (open3d.pipelines.registration.Feature): Source point cloud feature.
        target_feature (open3d.pipelines.registration.Feature): Target point cloud feature.
        mutual_filter (bool): Enables mutual filter such that the correspondence of the source point’s correspondence is itself.
        max_correspondence_distance (float): Maximum correspondence points-pair distance.
        estimation_method (open3d.pipelines.registration.TransformationEstimation, optional, default=TransformationEstimationPointToPoint without scaling.): Estimation method. One of (TransformationEstimationPointToPoint, TransformationEstimationPointToPlane, TransformationEstimationForGeneralizedICP, TransformationEstimationForColoredICP)
        ransac_n (int, optional, default=3): Fit ransac with ransac_n correspondences
        checkers (List[open3d.pipelines.registration.CorrespondenceChecker], optional, default=[]): Vector of Checker class to check if two point clouds can be aligned. One of (CorrespondenceCheckerBasedOnEdgeLength, CorrespondenceCheckerBasedOnDistance, CorrespondenceCheckerBasedOnNormal)
        criteria (open3d.pipelines.registration.RANSACConvergenceCriteria, optional, default=RANSACConvergenceCriteria class with max_iteration=100000, and confidence=9.990000e-01): Convergence criteria
        seed (Optional[int], optional, default=None): Random seed.

    Returns:
        open3d.pipelines.registration.RegistrationResult
    """
    ...
