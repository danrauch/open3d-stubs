from typing import Callable, Optional, overload, List, Union, Dict, Any, Set
from enum import Enum
from numpy import float32, float64
from numpy.typing import NDArray, ArrayLike
import open3d as o3d
from .. import geometry, camera
from . import rendering, gui


class SelectedIndex:
    index: int
    order: int
    point: ArrayLike
    def __init__(self, *args, **kwargs) -> None: ...


class SelectionPolygonVolume:
    axis_max: float
    axis_min: float
    bounding_polygon: NDArray[float64]
    orthogonal_axis: str
    def __init__(self, *args, **kwargs): ...
    def crop_point_cloud(self, input: geometry.PointCloud) -> geometry.PointCloud: ...


def draw(
    geometry=None,
    title: str = "Open3D",
    width: int = 1024,
    height: int = 768,
    actions=None,
    lookat: ArrayLike = None,
    eye: ArrayLike = None,
    up: ArrayLike = None,
    field_of_view: float = 60.0,
    bg_color=(1.0, 1.0, 1.0, 1.0),
    bg_image=None,
    ibl=None,
    ibl_intensity=None,
    show_skybox: bool = None,
    show_ui: bool = None,
    raw_mode: bool = False,
    point_size=None,
    line_width=None,
    animation_time_step=1.0,
    animation_duration=None,
    rpc_interface=False,
    on_init=None,
    on_animation_frame=None,
    on_animation_tick=None,
    non_blocking_and_return_uid=False,
) -> None: ...


@overload
def draw_geometries(
    geometry_list: List[geometry.Geometry],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
    point_show_normal: bool = False,
    mesh_show_wireframe: bool = False,
    mesh_show_back_face: bool = False,
) -> None: ...


@overload
def draw_geometries(
    geometry_list: List[geometry.Geometry],
    lookat: ArrayLike,
    up: ArrayLike,
    front: ArrayLike,
    zoom: float,
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
    point_show_normal: bool = False,
    mesh_show_wireframe: bool = False,
    mesh_show_back_face: bool = False,
) -> None: ...


def draw_geometries_with_animation_callback(
    geometry_list: List[geometry.Geometry],
    callback_function: Callable[[Visualizer], bool],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
) -> None: ...


def draw_geometries_with_custom_animation(
    geometry_list: List[geometry.Geometry],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
    optional_view_trajectory_json_file: str = "",
) -> None: ...


def draw_geometries_with_editing(
    geometry_list: List[geometry.Geometry],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
) -> None: ...


def draw_geometries_with_key_callbacks(
    geometry_list: List[geometry.Geometry],
    key_to_callback: Dict[int, Callable[[Visualizer], bool]],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
) -> None: ...


def draw_geometries_with_vertex_selection(
    geometry_list: List[geometry.Geometry],
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
) -> None: ...
def read_selection_polygon_volume(filename: str) -> SelectionPolygonVolume: ...


class ExternalVisualizer:
    def __init__(self, address=..., timeout=...) -> None: ...
    def set(self, obj=..., path=..., time=..., layer=..., connection=...) -> bool: ...
    def set_time(self, time): ...
    def set_active_camera(self, path): ...
    def draw(self, geometry=..., *args, **kwargs) -> None: ...


class MeshColorOption(Enum):
    Color = ...
    Default = ...
    Normal = ...
    XCoordinate = ...
    YCoordinate = ...
    ZCoordinate = ...


class MeshShadeOption(Enum):
    Color = ...
    Default = ...


class PickedPoint:
    coord: NDArray[float64]
    index: int
    def __init__(self) -> None: ...


class PointColorOption(Enum):
    Color = ...
    Default = ...
    Normal = ...
    XCoordinate = ...
    YCoordinate = ...
    ZCoordinate = ...


class RenderOption:
    background_color: NDArray[float64]
    light_on: bool
    line_width: float
    mesh_color_option: MeshColorOption
    mesh_shade_option: MeshShadeOption
    mesh_show_back_face: bool
    mesh_show_wireframe: bool
    point_color_option: PointColorOption
    point_show_normal: bool
    point_size: float
    show_coordinate_frame: bool
    def __init__(self) -> None: ...
    def load_from_json(self, filename: str) -> None: ...
    def save_to_json(self, filename: str) -> None: ...


class ViewControl:
    def __init__(self) -> None: ...

    def camera_local_rotate(
        self, x: float, y: float, xo: float = 0.0, yo: float = 0.0
    ) -> None: ...

    def camera_local_translate(
        self, forward: float, right: float, up: float
    ) -> None: ...
    def change_field_of_view(self, step: float = 0.45) -> None: ...

    def convert_from_pinhole_camera_parameters(
        self, parameter: camera.PinholeCameraParameters, allow_arbitrary: bool = False
    ) -> bool: ...

    def convert_to_pinhole_camera_parameters(
        self,
    ) -> camera.PinholeCameraParameters: ...
    def get_field_of_view(self) -> float: ...
    def reset_camera_local_rotate(self) -> None: ...
    def rotate(self, x: float, y: float, xo: float = 0.0, yo: float = 0.0) -> None: ...
    def scale(self, scale: float) -> None: ...
    def set_constant_z_far(self, z_far: float) -> None: ...
    def set_constant_z_near(self, z_near: float) -> None: ...
    def set_front(self, front: NDArray[float64]) -> None: ...
    def set_lookat(self, lookat: NDArray[float64]) -> None: ...
    def set_up(self, up: NDArray[float64]) -> None: ...
    def set_zoom(self, zoom: float) -> None: ...

    def translate(
        self, x: float, y: float, xo: float = 0.0, yo: float = 0.0
    ) -> None: ...
    def unset_constant_z_far(self) -> None: ...
    def unset_constant_z_near(self) -> None: ...


class Visualizer:
    def __init__(self) -> None: ...

    def add_geometry(
        self, geometry: geometry.Geometry, reset_bounding_box: bool = True
    ) -> bool: ...
    def capture_depth_float_buffer(self, do_render: bool = False) -> geometry.Image: ...

    def capture_depth_image(
        self, filename: str, do_render: bool = False, depth_scale: float = 1000.0
    ) -> None: ...

    def capture_depth_point_cloud(
        self,
        filename: str,
        do_render: bool = False,
        convert_to_world_coordinate: bool = False,
    ) -> None: ...

    def capture_screen_float_buffer(
        self, do_render: bool = False
    ) -> geometry.Image: ...
    def capture_screen_image(self, filename: str, do_render: bool = False) -> None: ...
    def clear_geometies(self) -> bool: ...
    def close(self) -> None: ...

    def create_window(
        self,
        window_name: str = "Open3D",
        width: int = 1920,
        height: int = 1080,
        left: int = 50,
        top: int = 50,
        visible: bool = True,
    ) -> bool: ...
    def destroy_window(self) -> None: ...
    def get_render_option(self) -> RenderOption: ...
    def get_view_control(self) -> ViewControl: ...
    def get_window_name(self) -> str: ...
    def is_full_screen(self) -> bool: ...
    def poll_events(self) -> bool: ...

    def register_animation_callback(
        self, callback_func: Callable[[Visualizer], bool]
    ) -> None: ...

    def remove_geometry(
        self, geometry: geometry.Geometry, reset_bounding_box: bool = True
    ) -> bool: ...
    def reset_view_point(self, reset_bounding_box: bool = False) -> None: ...
    def run(self) -> None: ...
    def set_full_screen(self, fullscreen: bool) -> None: ...
    def toggle_full_screen(self) -> None: ...
    def update_geometry(self, geometry: geometry.Geometry) -> bool: ...
    def update_renderer(self) -> None: ...


class VisualizerWithEditing(Visualizer):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self, voxel_size: float = -1.0, use_dialog: bool = True, directory: str = ""
    ) -> None: ...
    def get_picked_points(self) -> List[int]: ...


class VisualizerWithKeyCallback(Visualizer):
    def __init__(self) -> None: ...

    def register_key_action_callback(
        self, key: int, callback_func: Callable[[Visualizer, int, int], bool]
    ) -> None: ...

    def register_key_callback(
        self, key: int, callback_func: Callable[[Visualizer], bool]
    ) -> None: ...


class VisualizerWithVertexSelection(Visualizer):
    def __init__(self) -> None: ...
    def clear_picked_points(self) -> None: ...
    def get_picked_points(self) -> List[PickedPoint]: ...
    def register_selection_changed_callback(self, f: Callable[[], None]) -> None: ...
    def register_selection_moved_callback(self, f: Callable[[], None]) -> None: ...
    def register_selection_moving_callback(self, f: Callable[[], None]) -> None: ...


class O3DVisualizer(gui.Window):
    class DrawObject:
        geometry: o3d.geometry.Geometry3D
        group: str
        is_visible: bool
        name: str
        time: float
        def __init__(self, *args, **kwargs) -> None: ...

    class Shader(Enum):
        DEPTH = ...
        NORMALS = ...
        STANDARD = ...
        UNLIT = ...

    class TickResult(Enum):
        NO_CHANGE = ...
        REDRAW = ...

    def __init__(
        self, title: str = "Open3D", width: int = 1024, height: int = 768
    ) -> None: ...
    def add_3d_label(self, pos: NDArray[float32], text: str) -> None: ...

    def add_action(
        self, name: str, callback: Callable[[O3DVisualizer], None]
    ) -> None: ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: Union[o3d.geometry.Geometry, o3d.geometry.Geometry3D],
        material: Optional[rendering.MaterialRecord] = None,
        group: str = "",
        time: float = 0.0,
        is_visible: bool = True,
    ) -> None: ...

    @overload
    def add_geometry(
        self,
        name: str,
        geometry: o3d.t.geometry.Geometry,
        material: Optional[rendering.MaterialRecord] = None,
        group: str = "",
        time: float = 0.0,
        is_visible: bool = True,
    ) -> None: ...

    @overload
    def add_geometry(
        self,
        name: str,
        model: rendering.TriangleMeshModel,
        material: Optional[rendering.MaterialRecord] = None,
        group: str = "",
        time: float = 0.0,
        is_visible: bool = True,
    ) -> None: ...
    @overload
    def add_geometry(self, d: Dict[str, Any]) -> None: ...
    def clear_3d_labels(self) -> None: ...
    def close(self) -> None: ...
    def close_dialog(self) -> None: ...
    def enable_raw_mode(self, enable: bool) -> None: ...
    def export_current_image(self, path: str) -> None: ...
    def get_geometry(self, name: str) -> DrawObject: ...
    def get_geometry_material(self, name: str) -> rendering.MaterialRecord: ...
    def get_selection_sets(self) -> List[Dict[str, Set[SelectedIndex]]]: ...

    def modify_geometry_material(
        self, name: str, material: rendering.MaterialRecord
    ) -> None: ...
    def post_redraw(self) -> None: ...
    def remove_geometry(self, name: str) -> None: ...
    def reset_camera_to_default(self) -> None: ...

    def set_background(
        self, bg_color: NDArray[float32], bg_image: Optional[geometry.Image] = None
    ) -> None: ...
    def set_on_close(self, callback: Callable[[], bool]) -> None: ...

    @overload
    def setup_camera(
        self,
        fov: float,
        center: NDArray[float32],
        eye: NDArray[float32],
        up: NDArray[float32],
    ) -> None: ...

    @overload
    def setup_camera(
        self,
        intrinsic: camera.PinholeCameraIntrinsic,
        extrinsic: NDArray[float64],
    ) -> None: ...

    @overload
    def setup_camera(
        self,
        intrinsic: NDArray[float64],
        extrinsic: NDArray[float64],
        intrinsic_width_px: int,
        intrinsic_height_px: int,
    ) -> None: ...
