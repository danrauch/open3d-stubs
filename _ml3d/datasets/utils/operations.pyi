"""
This type stub file was generated by pyright.
"""


def create_3D_rotations(axis, angle):  # -> ndarray[Unknown, Unknown]:
    """Create rotation matrices from a list of axes and angles. Code from
    wikipedia on quaternions.

    Args:
        axis: float32[N, 3]
        angle: float32[N,]

    Returns:
        float32[N, 3, 3]
    """
    ...


def projection_matrix_to_CRT_kitti(proj):  # -> tuple[Unknown, Unknown, Unknown]:
    """Split projection matrix of kitti.

    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.

    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.

    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """
    ...


def get_frustum(bbox_image, C, near_clip=..., far_clip=...):
    """Get frustum corners in camera coordinates.

    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float): Nearest distance of frustum.
        far_clip (float): Farthest distance of frustum.

    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    ...


def camera_to_lidar(points, world_cam):
    """Convert points in camera coordinate to lidar coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        world_cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinates to lidar coordinates.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinates.
    """
    ...


def corners_nd(dims, origin=...):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ...


def rotation_3d_in_axis(points, angles, axis=...):  # -> Any:
    """Rotate points in specific axis.

    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.

    Returns:
        np.ndarray: Rotated points.
    """
    ...


def rotation_2d(points, angles):  # -> Any:
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    ...


def center_to_corner_box3d(centers, dims, angles=..., origin=...):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    ...


def center_to_corner_box2d(boxes, origin=...):  # -> Any:
    """Convert kitti locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    ...


def corner_to_surfaces_3d(corners):  # -> ndarray[Unknown, Unknown]:
    """Convert 3d box corners from corner function above to surfaces that normal
    vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    ...


def surface_equ_3d(polygon_surfaces):  # -> tuple[ndarray[Unknown, Unknown], Any]:
    """Compute normal vectors for polygon surfaces.

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    ...


def points_in_convex_polygon_3d(points, polygon_surfaces, num_surfaces=...):  # -> Any:
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of \
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3). \
            All surfaces' normal vector must direct to internal. \
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains \
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    ...


def points_in_box(points, rbbox, origin=..., camera_frame=..., cam_world=...):  # -> Any:
    """Check points in rotated bbox and return indices.

    If `rbbox` is in camera frame, it is first converted to world frame using
    `cam_world`. Returns a 2D array classifying each point for each box.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation (camera/world frame).
        origin (tuple[int]): Indicate the position of box center.
        camera_frame: True if `rbbox` are in camera frame(like kitti format, where y
          coordinate is height), False for [x, y, z, dx, dy, dz, yaw] format.
        cam_world: camera to world transformation matrix. Required when `camera_frame` is True.

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    ...


def filter_by_min_points(bboxes, min_points_dict):  # -> list[Unknown]:
    """Filter ground truths by number of points in the bbox."""
    ...


def random_sample(files, num):  # -> list[Unknown]:
    ...


def corner_to_standup_nd_jit(boxes_corner):  # -> ndarray[Unknown, Unknown]:
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    ...


def box_collision_test(boxes, qboxes):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
    """
    ...


def sample_class(class_name, num, gt_boxes, db_boxes):  # -> list[Unknown]:
    ...


def remove_points_in_boxes(points, boxes):
    """Remove the points in the sampled bounding boxes.

    Args:
        points (np.ndarray): Input point cloud array.
        boxes (np.ndarray): Sampled ground truth boxes.

    Returns:
        np.ndarray: Points with those in the boxes removed.
    """
    ...


def get_min_bbox(points):  # -> list[Unknown]:
    """Return minimum bounding box encapsulating points.

    Args:
        points (np.ndarray): Input point cloud array.

    Returns:
        np.ndarray: 3D BEV bounding box (x, y, z, w, h, l, yaw).
    """
    ...
