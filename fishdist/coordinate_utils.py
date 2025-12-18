# here I will store a set of wing tools

import numpy as np
from skimage.filters import threshold_yen
import affine6p
from batoolset.nps.tools import get_image_bounds
from batoolset.tools.logger import TA_logger  # logging

logger = TA_logger()  # logging_level=TA_logger.DEBUG

def remove_out_of_bonds_coords_nd(region_coords, bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim):
    """
    Removes coordinates that are outside of the given bounds.

    Args:
        region_coords (numpy.ndarray): The region coordinates.
        bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim (numpy.ndarray): Bounds array with min and max values for each dimension.

    Returns:
        numpy.ndarray: The region coordinates within the bounds.

    # Examples:
    #     Remove coordinates outside of the bounds:
    #     >>> new_coords = remove_out_of_bonds_coords_nd(region_coords, bounds_array)
    """
    for dim, axis_minb_max in enumerate(bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim):
        if region_coords[:, dim].min() < axis_minb_max[0] or region_coords[:, dim].max() >= axis_minb_max[1]:
            tmp = []
            for dim, axis_minb_max in enumerate(bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim):
                tmp.append(region_coords[:, dim] >= axis_minb_max[0])
                tmp.append(region_coords[:, dim] < axis_minb_max[1])
            final_stuff = np.logical_and.reduce(tmp)
            return region_coords[final_stuff]
    return region_coords


def remove_out_of_bonds_coords(region_coords, bounds):
    """
    Removes coordinates that are outside of the given bounds.

    Args:
        region_coords (numpy.ndarray): The region coordinates.
        bounds (tuple): The bounds of the region.

    Returns:
        numpy.ndarray: The region coordinates within the bounds.

    # Examples:
    #     Remove coordinates outside of the bounds:
    #     >>> new_coords = remove_out_of_bonds_coords(region_coords, bounds)
    """
    if region_coords[:, 0].min() < bounds[0] or region_coords[:, 0].max() >= bounds[2] or region_coords[:, 1].min() < bounds[1] or region_coords[:, 1].max() >= bounds[3]:
        return region_coords[((region_coords >= [bounds[0], bounds[1]]) & (region_coords < [bounds[2], bounds[3]])).all(axis=1)]
    else:
        return region_coords


def remove_out_of_image_coords(region_coords, img):
    """
    Removes coordinates that are outside of the image bounds.

    Args:
        region_coords (numpy.ndarray): The region coordinates.
        img (numpy.ndarray): The image.

    Returns:
        numpy.ndarray: The region coordinates within the image bounds.

    # Examples:
    #     Remove coordinates outside of the image bounds:
    #     >>> new_coords = remove_out_of_image_coords(region_coords, img)
    """
    if region_coords.min() < 0 or region_coords[:, 0].max() >= img.shape[0] or region_coords[:, 1].max() >= img.shape[1]:
        return region_coords[((region_coords >= [0, 0]) & (region_coords < [img.shape[0], img.shape[1]])).all(axis=1)]
    else:
        return region_coords


def make_point_in_bounds(point, bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim):
    """
    Adjusts a point to be within the given bounds.

    Args:
        point (numpy.ndarray): The point coordinates.
        bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim (numpy.ndarray): Bounds array with min and max values for each dimension.

    Returns:
        numpy.ndarray: The adjusted point within the bounds.

    # Examples:
    #     Adjust a point to be within the bounds:
    #     >>> new_point = make_point_in_bounds(point, bounds_array)
    """
    point = np.fmax(point, bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim[:, 0].T)
    point = np.fmin(point, bounds_array_as_n_and_2_dimensions_giving_min_and_max_of_each_dim[:, 1].T - 1)
    return point

def get_cuboid(image, centroid, cube_radius, return_cuboid_coords_instead=False, force_centroid_int=True):
    """
    Extracts a cuboid region from an image based on the centroid and cube radius.

    Args:
        image (numpy.ndarray): The input image.
        centroid (numpy.ndarray or list): The centroid coordinates.
        cube_radius (numpy.ndarray or list): The radius of the cuboid in each dimension.
        return_cuboid_coords_instead (bool, optional): If True, returns the coordinates of the cuboid instead of the cuboid itself. Defaults to False.
        force_centroid_int (bool, optional): If True, forces the centroid coordinates to be integers. Defaults to True.

    Returns:
        numpy.ndarray or list: The cuboid region or its coordinates.

    # Examples:
    #     Extract a cuboid region from an image:
    #     >>> cuboid = get_cuboid(image, centroid, cube_radius)
    #
    #     Extract the coordinates of a cuboid region:
    #     >>> cuboid_coords = get_cuboid(image, centroid, cube_radius, return_cuboid_coords_instead=True)
    """
    if centroid is None or image is None or cube_radius is None:
        return None

    if len(cube_radius) != len(image.shape):
        logger.error('dimension mismatch between cube and image dimensions, cuboid cannot be recovered')
        return None

    if not isinstance(cube_radius, np.ndarray):
        centroid = np.asarray(centroid)

    if not isinstance(cube_radius, np.ndarray):
        cube_radius = np.asarray(cube_radius)

    if force_centroid_int:
        centroid = np.rint(centroid).astype(int)

    begin = centroid - cube_radius
    end = centroid + cube_radius

    bounds = get_image_bounds(image)
    begin = make_point_in_bounds(begin, bounds)
    end = make_point_in_bounds(end, bounds)

    cuboid_bounds = np.stack([begin, end + 1], axis=0)

    slices = []
    for begin, end in cuboid_bounds.T:
        slices.append(slice(begin, end))

    if return_cuboid_coords_instead:
        return convert_slices_to_set_of_coordinates(slices)

    return image[tuple(slices)]


def nd_slice_to_indexes(nd_slice):
    """
    Converts an n-dimensional slice to indices.

    Args:
        nd_slice (slice or tuple): The n-dimensional slice.

    Returns:
        tuple: Indexes for each dimension.

    # Examples:
    #     Convert an n-dimensional slice to indices:
    #     >>> indexes = nd_slice_to_indexes(nd_slice)
    """
    grid = np.mgrid[{tuple: nd_slice, slice: (nd_slice,)}[type(nd_slice)]]
    return tuple(grid[i].ravel() for i in range(grid.shape[0]))


def convert_slices_to_set_of_coordinates(slices_list):
    """
    Converts a list of slices to a set of coordinates.

    Args:
        slices_list (list): List of slices.

    Returns:
        numpy.ndarray: Array of coordinates.

    # Examples:
    #     Convert a list of slices to a set of coordinates:
    #     >>> coordinates = convert_slices_to_set_of_coordinates(slices_list)
    """
    arrays = np.stack(nd_slice_to_indexes(tuple(slices_list)), axis=0)
    arrays = arrays.T
    return arrays



def compute_registration_matrix(anterior_src: np.ndarray, posterior_src: np.ndarray,
                                proximal_src: np.ndarray, distal_src: np.ndarray,
                                anterior_target: np.ndarray, posterior_target: np.ndarray,
                                proximal_target: np.ndarray, distal_target: np.ndarray,
                                return_transformer: bool = False) -> np.ndarray:
    """
    Compute the registration matrix for alignment.

    Args:
        anterior_src: Source image of the anterior landmarks.
        posterior_src: Source image of the posterior landmarks.
        proximal_src: Source image of the proximal landmarks.
        distal_src: Source image of the distal landmarks.
        anterior_target: Target image of the anterior landmarks.
        posterior_target: Target image of the posterior landmarks.
        proximal_target: Target image of the proximal landmarks.
        distal_target: Target image of the distal landmarks.
        return_transformer: Flag indicating whether to return the affine transformation object.

    Returns:
        Registration matrix as a numpy array or affine transformation object.

    """

    src = [anterior_src, posterior_src, proximal_src, distal_src]
    target = [anterior_target, posterior_target, proximal_target, distal_target]
    trans = affine6p.estimate(src, target)
    if return_transformer:
        return trans
    else:
        return trans.get_matrix()
