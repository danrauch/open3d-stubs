from enum import Enum
from open3d import geometry, pipelines, camera, visualization


def read_point_cloud(
    filename: str,
    format: str = "auto",
    remove_nan_points: bool = True,
    remove_infinite_points: bool = True,
    print_progess: bool = False,
) -> geometry.PointCloud: ...


def write_point_cloud(
    filename: str,
    pointcloud: geometry.PointCloud,
    write_ascii: bool = False,
    compressed: bool = False,
    print_progess: bool = False,
) -> bool: ...
def read_feature(filename: str) -> pipelines.registration.Feature: ...
def read_file_geometry_type(path: str) -> FileGeometry: ...
def read_image(filename: str) -> geometry.Image: ...


def read_line_set(
    filename: str, format: str = "auto", print_progess: bool = False
) -> geometry.LineSet: ...
def read_pinhole_camera_intrinsic(filename: str) -> camera.PinholeCameraIntrinsic: ...
def read_pinhole_camera_parameters(filename: str) -> camera.PinholeCameraParameters: ...
def read_pinhole_camera_trajectory(filename: str) -> camera.PinholeCameraTrajectory: ...
def read_pose_graph(filename: str) -> pipelines.registration.PoseGraph: ...


def read_triangle_mesh(
    filename: str, enable_post_processing: bool = False, print_progess: bool = False
) -> geometry.TriangleMesh: ...


def read_triangle_model(
    filename: str, print_progess: bool = False
) -> visualization.rendering.TriangleMeshModel: ...


def read_voxel_grid(
    filename: str, format: str = "auto", print_progess: bool = False
) -> geometry.VoxelGrid: ...
def write_feature(filename: str, feature: pipelines.registration.Feature) -> bool: ...
def write_image(filename: str, image: geometry.Image, quality: int = -1) -> bool: ...


def write_line_set(
    filename: str,
    line_set: geometry.LineSet,
    write_ascii: bool = False,
    compressed: bool = False,
    print_progess: bool = False,
) -> bool: ...


def write_pinhole_camera_intrinsic(
    filename: str, intrinsic: camera.PinholeCameraIntrinsic
) -> bool: ...


def write_pinhole_camera_parameters(
    filename: str, parameters: camera.PinholeCameraParameters
) -> bool: ...


def write_pinhole_camera_trajectory(
    filename: str, trajectory: camera.PinholeCameraTrajectory
) -> bool: ...


def write_pose_graph(
    filename: str, pose_graph: pipelines.registration.PoseGraph
) -> bool: ...


def write_triangle_mesh(
    filename: str,
    mesh: geometry.TriangleMesh,
    write_ascii: bool = False,
    compressed: bool = False,
    write_vertex_normals: bool = True,
    write_vertex_colors: bool = True,
    write_triangle_uvs: bool = True,
    print_progess: bool = False,
) -> bool: ...


def write_voxel_grid(
    filename: str,
    voxel_grid: geometry.VoxelGrid,
    write_ascii: bool = False,
    compressed: bool = False,
    print_progess: bool = False,
) -> bool: ...


class FileGeometry(Enum):
    CONTAINS_LINES = ...
    CONTAINS_POINTS = ...
    CONTAINS_TRIANGLES = ...
    CONTENTS_UNKNOWN = ...
