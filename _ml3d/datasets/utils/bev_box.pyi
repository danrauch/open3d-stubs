"""
This type stub file was generated by pyright.
"""

from ...vis import BoundingBox3D


class BEVBox3D(BoundingBox3D):
    """Class that defines a special bounding box for object detection, with only
    one rotation axis (yaw).

                            up z    x front (yaw=0.5*pi)
                                ^   ^
                                |  /
                                | /
        (yaw=pi) left y <------ 0

    The relative coordinate of bottom center in a BEV box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the negative direction of y axis, and increases from
    the negative direction of y to the positive direction of x.
    """

    def __init__(self, center, size, yaw, label_class, confidence, world_cam=..., cam_img=..., **kwargs) -> None:
        """Creates a bounding box.

        Args:
            center: (x, y, z) that defines the center of the box
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge
            yaw: yaw angle of box
            label_class: integer specifying the classification label. If an LUT is
                specified in create_lines() this will be used to determine the color
                of the box.
            confidence: confidence level of the box
            world_cam: world to camera transformation
            cam_img: camera to image transformation
        """
        ...

    def to_kitti_format(self, score=...):  # -> str:
        """This method transforms the class to KITTI format."""
        ...

    def generate_corners3d(self):
        """Generate corners3d representation for this object.

        Returns:
            corners_3d: (8, 3) corners of box3d in camera coordinates.
        """
        ...

    def to_xyzwhlr(self):  # -> ndarray[Unknown, Unknown]:
        """Returns box in the common 7-sized vector representation: (x, y, z, w,
        l, h, a), where (x, y, z) is the bottom center of the box, (w, l, h) is
        the width, length and height of the box a is the yaw angle.

        Returns:
            box: (7,)
        """
        ...

    def to_camera(self):  # -> Any | ndarray[Unknown, Unknown]:
        """Transforms box into camera space.

                     up x    y front
                        ^   ^
                        |  /
                        | /
         left z <------ 0

        Returns box in the common 7-sized vector representation:
        (x, y, z, l, h, w, a), where
        (x, y, z) is the bottom center of the box,
        (l, h, w) is the length, height, width of the box
        a is the yaw angle

        Returns:
            transformed box: (7,)
        """
        ...

    def to_img(self):  # -> None:
        """Transforms box into 2d box.

        Returns:
            transformed box: (4,)
        """
        ...

    def get_difficulty(self):  # -> int:
        """General method to compute difficulty, can be overloaded.

        Returns:
            Difficulty depending on projected height of box.
        """
        ...

    def to_dict(self):  # -> dict[str, Unknown]:
        """Convert data for evaluation:"""
        ...

    @staticmethod
    def to_dicts(bboxes):  # -> dict[str, ndarray[Unknown, Unknown]]:
        """Convert data for evaluation:

        Args:
            bboxes: List of BEVBox3D bboxes.
        """
        ...
