"""
This type stub file was generated by pyright.
"""

from .base_dataset import BaseDataset, BaseDatasetSplit

log = ...


class SemanticKITTI(BaseDataset):
    """This class is used to create a dataset based on the SemanticKitti
    dataset, and used in visualizer, training, or testing.

    The dataset is best for semantic scene understanding.
    """

    def __init__(self, dataset_path, name=..., cache_dir=..., use_cache=..., class_weights=..., ignored_label_inds=..., test_result_folder=..., test_split=..., training_split=..., validation_split=..., all_split=..., **kwargs) -> None:
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

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

    def get_split(self, split):  # -> SemanticKITTISplit:
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        ...

    def is_tested(self, attr):  # -> bool:
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        ...

    def save_test_result(self, results, attr):  # -> None:
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        ...

    def save_test_result_kpconv(self, results, inputs):  # -> None:
        ...

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        ...


class SemanticKITTISplit(BaseDatasetSplit):
    def __init__(self, dataset, split=...) -> None:
        ...

    def __len__(self):  # -> int:
        ...

    def get_data(self, idx):  # -> dict[str, Unknown]:
        ...

    def get_attr(self, idx):  # -> dict[str, Unknown]:
        ...
