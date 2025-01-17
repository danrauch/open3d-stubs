"""
This type stub file was generated by pyright.
"""


class BoundingBox3D:
    """Class that defines an axially-oriented bounding box."""
    next_id = ...

    def __init__(self, center, front, up, left, size, label_class, confidence, meta=..., show_class=..., show_confidence=..., show_meta=..., identifier=..., arrow_length=...) -> None:
        """Creates a bounding box.

        Front, up, left define the axis of the box and must be normalized and
        mutually orthogonal.

        Args:
            center: (x, y, z) that defines the center of the box
            front: normalized (i, j, k) that defines the front direction of the box
            up: normalized (i, j, k) that defines the up direction of the box
            left: normalized (i, j, k) that defines the left direction of the box
            size: (width, height, depth) that defines the size of the box, as
                measured from edge to edge
            label_class: integer specifying the classification label. If an LUT is
                specified in create_lines() this will be used to determine the color
                of the box.
            confidence: confidence level of the box
            meta: a user-defined string (optional)
            show_class: displays the class label in text near the box (optional)
            show_confidence: displays the confidence value in text near the box
                (optional)
            show_meta: displays the meta string in text near the box (optional)
            identifier: a unique integer that defines the id for the box (optional,
                will be generated if not provided)
            arrow_length: the length of the arrow in the front_direct. Set to zero
                to disable the arrow (optional)
        """
        ...

    def __repr__(self):  # -> str:
        ...

    @staticmethod
    def create_lines(boxes, lut=...):
        """Creates and returns an open3d.geometry.LineSet that can be used to
        render the boxes.

        Args:
            boxes: the list of bounding boxes
            lut: a ml3d.vis.LabelLUT that is used to look up the color based on
                the label_class argument of the BoundingBox3D constructor. If
                not provided, a color of 50% grey will be used. (optional)
        """
        ...
