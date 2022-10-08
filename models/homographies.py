import tensorflow as tf

# from tensorflow.contrib.image import transform as H_transform
from tensorflow_addons.image import transform as H_transform
from math import pi
import cv2 as cv

from utils.tools import dict_update


homography_adaptation_default_config = {
    "num": 1,
    "aggregation": "sum",
    "homographies": {
        "translation": True,
        "rotation": True,
        "scaling": True,
        "perspective": True,
        "scaling_amplitude": 0.1,
        "perspective_amplitude_x": 0.1,
        "perspective_amplitude_y": 0.1,
        "patch_ratio": 0.5,
        "max_angle": pi,
    },
    "filter_counts": 0,
}


def homography_adaptation(image, net, config):
    """Perfoms homography adaptation.
    Inference using multiple random warped patches of the same input image for robust
    predictions.
    Arguments:
        image: A `Tensor` with shape `[N, H, W, 1]`.
        net: A function that takes an image as input, performs inference, and outputs the
            prediction dictionary.
        config: A configuration dictionary containing optional entries such as the number
            of sampled homographies `'num'`, the aggregation method `'aggregation'`.
    Returns:
        A dictionary which contains the aggregated detection probabilities.
    """

    probs = net(image)["prob"]
    counts = tf.ones_like(probs)
    images = image

    probs = tf.expand_dims(probs, axis=-1)
    counts = tf.expand_dims(counts, axis=-1)
    images = tf.expand_dims(images, axis=-1)

    shape = tf.shape(input=image)[1:3]
    config = dict_update(homography_adaptation_default_config, config)

    def step(i, probs, counts, images):
        # Sample image patch
        H = sample_homography(shape, **config["homographies"])
        H_inv = invert_homography(H)
        warped = H_transform(image, H, interpolation="BILINEAR")
        count = H_transform(tf.expand_dims(tf.ones(tf.shape(input=image)[:3]), -1), H_inv, interpolation="NEAREST")[..., 0]

        # Predict detection probabilities
        warped_shape = tf.cast(tf.cast(shape, dtype=tf.float32) * config["homographies"]["patch_ratio"], dtype=tf.int32)
        input_warped = tf.image.resize(warped, warped_shape)
        prob = net(input_warped)["prob"]
        prob = tf.image.resize(tf.expand_dims(prob, axis=-1), shape)[..., 0]
        prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv, interpolation="BILINEAR")[..., 0]

        probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
        counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
        images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
        return i + 1, probs, counts, images

    _, probs, counts, images = tf.while_loop(
        cond=lambda i, p, c, im: tf.less(i, config["num"] - 1),
        body=step,
        loop_vars=[0, probs, counts, images],
        parallel_iterations=1,
        back_prop=False,
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape([None, None, None, 1, None]),
        ],
    )

    counts = tf.reduce_sum(input_tensor=counts, axis=-1)
    max_prob = tf.reduce_max(input_tensor=probs, axis=-1)
    mean_prob = tf.reduce_sum(input_tensor=probs, axis=-1) / counts

    if config["aggregation"] == "max":
        prob = max_prob
    elif config["aggregation"] == "sum":
        prob = mean_prob
    else:
        raise ValueError("Unkown aggregation method: {}".format(config["aggregation"]))

    if config["filter_counts"]:
        prob = tf.compat.v1.where(tf.greater_equal(counts, config["filter_counts"]), prob, tf.zeros_like(prob))

    return {"prob": prob, "counts": counts, "mean_prob": mean_prob, "input_images": images, "H_probs": probs}  # debug


def sample_homography(
    shape,
    perspective=True,
    scaling=True,
    rotation=True,
    translation=True,
    n_scales=5,
    n_angles=25,
    scaling_amplitude=0.1,
    perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1,
    patch_ratio=0.5,
    max_angle=pi / 2,
    allow_artifacts=False,
    translation_overflow=0.0,
):
    """Sample a random valid homography.

    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = tf.stack([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + tf.constant([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], tf.float32)

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = tf.random.truncated_normal([1], 0.0, perspective_amplitude_y / 2)
        h_displacement_left = tf.random.truncated_normal([1], 0.0, perspective_amplitude_x / 2)
        h_displacement_right = tf.random.truncated_normal([1], 0.0, perspective_amplitude_x / 2)
        pts2 += tf.stack(
            [
                tf.concat([h_displacement_left, perspective_displacement], 0),
                tf.concat([h_displacement_left, -perspective_displacement], 0),
                tf.concat([h_displacement_right, perspective_displacement], 0),
                tf.concat([h_displacement_right, -perspective_displacement], 0),
            ]
        )

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = tf.concat([[1.0], tf.random.truncated_normal([n_scales], 1, scaling_amplitude / 2)], 0)
        center = tf.reduce_mean(input_tensor=pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(tf.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = tf.range(n_scales)  # all scales are valid except scale=1
        else:
            valid = tf.compat.v1.where(tf.reduce_all(input_tensor=(scaled >= 0.0) & (scaled < 1.0), axis=[1, 2]))[:, 0]
        idx = valid[tf.random.uniform((), maxval=tf.shape(input=valid)[0], dtype=tf.int32)]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = tf.reduce_min(input_tensor=pts2, axis=0), tf.reduce_min(input_tensor=1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += tf.expand_dims(
            tf.stack([tf.random.uniform((), -t_min[0], t_max[0]), tf.random.uniform((), -t_min[1], t_max[1])]), axis=0
        )

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = tf.linspace(tf.constant(-max_angle), tf.constant(max_angle), n_angles)
        angles = tf.concat([angles, [0.0]], axis=0)  # in case no rotation is valid
        center = tf.reduce_mean(input_tensor=pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles), tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(tf.tile(tf.expand_dims(pts2 - center, axis=0), [n_angles + 1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = tf.range(n_angles)  # all angles are valid, except angle=0
        else:
            valid = tf.compat.v1.where(tf.reduce_all(input_tensor=(rotated >= 0.0) & (rotated < 1.0), axis=[1, 2]))[:, 0]
        idx = valid[tf.random.uniform((), maxval=tf.shape(input=valid)[0], dtype=tf.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = tf.cast(shape[::-1], dtype=tf.float32)  # different convention [y, x]
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(a=tf.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = tf.transpose(a=tf.linalg.lstsq(a_mat, p_mat, fast=True))
    return homography


def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return mat2flat(tf.linalg.inv(flat2mat(H)))


def flat2mat(H):
    """
    Converts a flattened homography transformation with shape `[1, 8]` to its
    corresponding homography matrix with shape `[1, 3, 3]`.
    """
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(input=H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    """
    Converts an homography matrix with shape `[1, 3, 3]` to its corresponding flattened
    homography transformation with shape `[1, 8]`.
    """
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    mask = H_transform(tf.ones(image_shape), homography, interpolation="NEAREST")
    if erosion_radius > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        mask = (
            tf.nn.erosion2d(
                mask[tf.newaxis, ..., tf.newaxis],
                tf.cast(tf.constant(kernel)[..., tf.newaxis], dtype=tf.float32),
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                "SAME",
                data_format="NHWC",
            )[0, ..., 0]
            + 1.0
        )
    return tf.cast(mask, dtype=tf.int32)


def warp_points(points, homography):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform

    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 8) and (8,) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    H = tf.expand_dims(homography, axis=0) if len(homography.shape) == 1 else homography

    # Get the points to the homogeneous format
    num_points = tf.shape(input=points)[0]
    points = tf.cast(points, tf.float32)[:, ::-1]
    points = tf.concat([points, tf.ones([num_points, 1], dtype=tf.float32)], -1)

    # Apply the homography
    H_inv = tf.transpose(a=flat2mat(invert_homography(H)))
    warped_points = tf.tensordot(points, H_inv, [[1], [0]])
    warped_points = warped_points[:, :2, :] / warped_points[:, 2:, :]
    warped_points = tf.transpose(a=warped_points, perm=[2, 0, 1])[:, :, ::-1]

    return warped_points[0] if len(homography.shape) == 1 else warped_points


def filter_points(points, shape):
    with tf.compat.v1.name_scope("filter_points"):
        mask = (points >= 0) & (points <= tf.cast(shape - 1, dtype=tf.float32))
        return tf.boolean_mask(tensor=points, mask=tf.reduce_all(input_tensor=mask, axis=-1))


# TODO: cleanup the two following functions
def warp_keypoints_to_list(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform

    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)

    Returns: a Tensor of size (num_keypoints, 2) with the new coordinates
             of the warped keypoints.
    """
    keypoints_map = packed_arg[0]
    H = packed_arg[1]
    if len(H.shape.as_list()) < 2:
        H = tf.expand_dims(H, 0)  # add a batch of 1
    # Get the keypoints list in homogeneous format
    keypoints = tf.cast(tf.compat.v1.where(keypoints_map > 0), tf.float32)
    keypoints = keypoints[:, ::-1]
    n_keypoints = tf.shape(input=keypoints)[0]
    keypoints = tf.concat([keypoints, tf.ones([n_keypoints, 1], dtype=tf.float32)], 1)

    # Apply the homography
    H_inv = invert_homography(H)
    H_inv = flat2mat(H_inv)
    H_inv = tf.transpose(a=H_inv[0, ...])
    warped_keypoints = tf.matmul(keypoints, H_inv)
    warped_keypoints = tf.round(warped_keypoints[:, :2] / warped_keypoints[:, 2:])
    warped_keypoints = warped_keypoints[:, ::-1]

    return warped_keypoints


def warp_keypoints_to_map(packed_arg):
    """
    Warp a map of keypoints (pixel is 1 for a keypoint and 0 else) with
    the INVERSE of the homography H.
    The inverse is used to be coherent with tf.contrib.image.transform

    Arguments:
        packed_arg: a tuple equal to (keypoints_map, H)

    Returns: a map of keypoints of the same size as the original keypoint_map.
    """
    warped_keypoints = tf.cast(warp_keypoints_to_list(packed_arg), dtype=tf.int32)
    n_keypoints = tf.shape(input=warped_keypoints)[0]
    shape = tf.shape(input=packed_arg[0])

    # Remove points outside the image
    zeros = tf.cast(tf.zeros([n_keypoints]), dtype=tf.bool)
    ones = tf.cast(tf.ones([n_keypoints]), dtype=tf.bool)
    loc = tf.logical_and(
        tf.compat.v1.where(warped_keypoints[:, 0] >= 0, ones, zeros),
        tf.compat.v1.where(warped_keypoints[:, 0] < shape[0], ones, zeros),
    )
    loc = tf.logical_and(loc, tf.compat.v1.where(warped_keypoints[:, 1] >= 0, ones, zeros))
    loc = tf.logical_and(loc, tf.compat.v1.where(warped_keypoints[:, 1] < shape[1], ones, zeros))
    warped_keypoints = tf.boolean_mask(tensor=warped_keypoints, mask=loc)

    # Output the new map of keypoints
    new_map = tf.scatter_nd(warped_keypoints, tf.ones([tf.shape(input=warped_keypoints)[0]], dtype=tf.float32), shape)

    return new_map
