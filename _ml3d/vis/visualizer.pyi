"""
This type stub file was generated by pyright.
"""

from .boundingbox import *
from .colormap import *
from .labellut import *


class Model:
    """The class that helps build visualization models based on attributes,
    data, and methods.
    """
    bounding_box_prefix = ...

    class BoundingBoxData:
        """The class to define a bounding box that is used to describe the
        target location.

        Args:
            name: The name of the pointcloud array.
            boxes: The array of pointcloud that define the bounding box.
        """

        def __init__(self, name, boxes) -> None:
            ...

    def __init__(self) -> None:
        ...

    def is_loaded(self, name):  # -> bool:
        """Check if the data is loaded."""
        ...

    def load(self, name, fail_if_no_space=...):  # -> NoReturn:
        """If data is not loaded, then load the data."""
        ...

    def unload(self, name):  # -> NoReturn:
        ...

    def create_point_cloud(self, data):  # -> None:
        """Create a point cloud based on the data provided.

        The data should include name and points.
        """
        ...

    def get_attr(self, name, attr_name):  # -> None:
        """Get an attribute from data based on the name passed."""
        ...

    def get_attr_shape(self, name, attr_name):  # -> list[Unknown]:
        """Get a shape from data based on the name passed."""
        ...

    def get_attr_minmax(self, attr_name, channel):  # -> tuple[float, float]:
        """Get the minimum and maximum for an attribute."""
        ...

    def get_available_attrs(self, names):  # -> List[Unknown]:
        """Get a list of attributes based on the name."""
        ...

    def calc_bounds_for(self, name):  # -> list[tuple[Unknown, Unknown, Unknown]] | list[tuple[float, float, float]]:
        """Calculate the bounds for a pointcloud."""
        ...


class DataModel(Model):
    """The class for data i/o and storage of visualization.

    Args:
        userdata: The dataset to be used in the visualization.
    """

    def __init__(self, userdata) -> None:
        ...

    def load(self, name, fail_if_no_space=...):  # -> None:
        """Load a pointcloud based on the name provided."""
        ...

    def unload(self, name):  # -> None:
        """Unload a pointcloud."""
        ...


class DatasetModel(Model):
    """The class used to manage a dataset model.

    Args:
        dataset:  The 3D ML dataset to use. You can use the base dataset, sample datasets , or a custom dataset.
        split: A string identifying the dataset split that is usually one of 'training', 'test', 'validation', or 'all'.
        indices: The indices to be used for the datamodel. This may vary based on the split used.
    """

    def __init__(self, dataset, split, indices) -> None:
        ...

    def is_loaded(self, name):  # -> bool:
        """Check if the data is loaded."""
        ...

    def load(self, name, fail_if_no_space=...):  # -> bool:
        """Check if data is not loaded, and then load the data."""
        ...

    def unload(self, name):  # -> None:
        """Unload the data (if it was loaded earlier)."""
        ...


class Visualizer:
    """The visualizer class for dataset objects and custom point clouds."""
    class LabelLUTEdit:
        """This class includes functionality for managing a labellut (label
        look-up-table).
        """

        def __init__(self) -> None:
            ...

        def clear(self):  # -> None:
            """Clears the look-up table."""
            ...

        def is_empty(self):  # -> bool:
            """Checks if the look-up table is empty."""
            ...

        def get_colors(self):  # -> list[Unknown]:
            """Returns a list of label keys."""
            ...

        def set_on_changed(self, callback):  # -> None:
            ...

        def set_labels(self, labellut):  # -> None:
            """Updates the labels based on look-up table passsed."""
            ...

    class ColormapEdit:
        """This class is used to create a color map for visualization of
        points.
        """

        def __init__(self, window, em) -> None:
            ...

        def set_on_changed(self, callback):  # -> None:
            ...

        def update(self, colormap, min_val, max_val):  # -> None:
            """Updates the colormap based on the minimum and maximum values
            passed.
            """
            ...

    class ProgressDialog:
        """This class is used to manage the progress dialog displayed during
        visualization.

        Args:
            title: The title of the dialog box.
            window: The window where the progress dialog box should be displayed.
            n_items: The maximum number of items.
        """

        def __init__(self, title, window, n_items) -> None:
            ...

        def set_text(self, text):  # -> None:
            """Set the label text on the dialog box."""
            ...

        def post_update(self, text=...):  # -> None:
            """Post updates to the main thread."""
            ...

        def update(self):  # -> None:
            """Enumerate the progress in the dialog box."""
            ...

    SOLID_NAME = ...
    LABELS_NAME = ...
    RAINBOW_NAME = ...
    GREYSCALE_NAME = ...
    COLOR_NAME = ...
    X_ATTR_NAME = ...
    Y_ATTR_NAME = ...
    Z_ATTR_NAME = ...

    def __init__(self) -> None:
        ...

    def set_lut(self, attr_name, lut):  # -> None:
        """Set the LUT for a specific attribute.

        Args:
        attr_name: The attribute name as string.
        lut: The LabelLUT object that should be updated.
        """
        ...

    def setup_camera(self):  # -> None:
        """Set up camera for visualization."""
        ...

    def show_geometries_under(self, name, show):  # -> None:
        """Show geometry for a given node."""
        ...

    def visualize_dataset(self, dataset, split, indices=..., width=..., height=...):  # -> None:
        """Visualize a dataset.

        Example:
            Minimal example for visualizing a dataset::
                import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

                dataset = ml3d.datasets.SemanticKITTI(dataset_path='/path/to/SemanticKITTI/')
                vis = ml3d.vis.Visualizer()
                vis.visualize_dataset(dataset, 'all', indices=range(100))

        Args:
            dataset: The dataset to use for visualization.
            split: The dataset split to be used, such as 'training'
            indices: An iterable with a subset of the data points to visualize, such as [0,2,3,4].
            width: The width of the visualization window.
            height: The height of the visualization window.
        """
        ...

    def visualize(self, data, lut=..., bounding_boxes=..., width=..., height=...):  # -> None:
        """Visualize a custom point cloud data.

        Example:
            Minimal example for visualizing a single point cloud with an
            attribute::

                import numpy as np
                import open3d.ml.torch as ml3d
                # or import open3d.ml.tf as ml3d

                data = [ {
                    'name': 'my_point_cloud',
                    'points': np.random.rand(100,3).astype(np.float32),
                    'point_attr1': np.random.rand(100).astype(np.float32),
                    } ]

                vis = ml3d.vis.Visualizer()
                vis.visualize(data)

        Args:
            data: A list of dictionaries. Each dictionary is a point cloud with
                attributes. Each dictionary must have the entries 'name' and
                'points'. Points and point attributes can be passed as numpy
                arrays, PyTorch tensors or TensorFlow tensors.
            lut: Optional lookup table for colors.
            bounding_boxes: Optional bounding boxes.
            width: window width.
            height: window height.
        """
        ...
