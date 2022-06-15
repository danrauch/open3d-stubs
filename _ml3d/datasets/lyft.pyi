"""
This type stub file was generated by pyright.
"""

from .base_dataset import BaseDataset

log = ...


class Lyft(BaseDataset):
    """This class is used to create a dataset based on the Lyft dataset, and
    used in object detection, visualizer, training, or testing.

    The Lyft level 5 dataset is best suited for self-driving applications.
    """

    def __init__(self, dataset_path, info_path=..., name=..., cache_dir=..., use_cache=..., **kwargs) -> None:
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about
            the dataset. This is default to dataset path if nothing is
            provided.
            name: The name of the dataset (Lyft in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
        ...

    @staticmethod
    def get_label_to_names():  # -> dict[int, str]:
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        ...

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        ...

    @staticmethod
    def read_label(info, calib):  # -> list[Unknown]:
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        ...

    def get_split(self, split):  # -> LyftSplit:
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        ...

    def get_split_list(self, split):  # -> Any | dict[Unknown, Unknown]:
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        ...

    def is_tested():  # -> None:
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the attribute is stored; else, returns false.
        """
        ...

    def save_test_result():  # -> None:
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        ...


class LyftSplit:
    def __init__(self, dataset, split=...) -> None:
        ...

    def __len__(self):  # -> int:
        ...

    def get_data(self, idx):  # -> dict[str, Unknown]:
        ...

    def get_attr(self, idx):  # -> dict[str, Unknown]:
        ...
