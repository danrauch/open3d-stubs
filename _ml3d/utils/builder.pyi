"""
This type stub file was generated by pyright.
"""

MODEL = ...
DATASET = ...
PIPELINE = ...
SAMPLER = ...


def build(cfg, registry, args=...):
    ...


def build_network(cfg):
    ...


def convert_device_name(framework):  # -> Literal['cuda', 'cpu']:
    """Convert device to either cpu or cuda."""
    ...


def convert_framework_name(framework):  # -> Literal['tf', 'torch']:
    """Convert framework to either tf or torch."""
    ...


def get_module(module_type, module_name, framework=..., **kwargs):
    """Fetch modules (pipeline, model, or) from registry."""
    ...
