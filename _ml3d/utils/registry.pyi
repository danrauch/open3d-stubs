"""
This type stub file was generated by pyright.
"""


class Registry:
    def __init__(self, name) -> None:
        ...

    def get(self, key, framework):  # -> None:
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        ...

    @property
    def name(self):  # -> Unknown:
        ...

    @property
    def module_dict(self):  # -> dict[Unknown, Unknown]:
        ...

    def register_module(self, framework=..., name=...):  # -> (cls: Unknown) -> None:
        ...


def get_from_name(module_name, registry, framework):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    ...
