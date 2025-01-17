"""
This type stub file was generated by pyright.
"""


class SemSegSpatiallyRegularSampler:
    """Spatially regularSampler sampler for semantic segmentation datsets."""

    def __init__(self, dataset) -> None:
        ...

    def __len__(self):  # -> int:
        ...

    def initialize_with_dataloader(self, dataloader):  # -> None:
        ...

    def get_cloud_sampler(self):  # -> Generator[int, None, None]:
        ...

    # -> (patchwise: Unknown = True, **kwargs: Unknown) -> tuple[Unknown, Unbound | Unknown | ndarray[Unknown, Unknown], Unbound | Unknown] | None:
    def get_point_sampler(self):
        ...
