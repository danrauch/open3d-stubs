

class Dataset:
    data_root: str
    download_dir: str
    extract_dir: str
    prefix: str
    def __init__(self, prefix: str, data_root: str = '') -> None: ...


class BunnyMesh(Dataset):
    """Data class for BunnyMesh contains the BunnyMesh.ply from the Stanford 3D Scanning Repository."""
    path: str
    def __init__(self, data_root: str = ...) -> None: ...


class ArmadilloMesh(Dataset):
    """Data class for ArmadilloMesh contains the ArmadilloMesh.ply from the Stanford 3D Scanning Repository."""
    path: str
    def __init__(self, data_root: str = ...) -> None: ...


class EaglePointCloud(Dataset):
    """Data class for EaglePointCloud contains the EaglePointCloud.ply file."""
    path: str
    def __init__(self, data_root: str = ...) -> None: ...
