# to keep in mind ZEISS 880 resolution: xy 140nm / z 400nm at 488nm

import os.path
import traceback
from scipy.ndimage import generate_binary_structure, grey_dilation, gaussian_laplace
from batoolset.SQLites.tools import get_column_from_sqlite_table, set_voxel_size, get_tables, drop_table_if_exists
from string_tools import levenshtein_distance, _extract_base_name, longest_common_substring
from batoolset.utils.loadlist import loadlist
from spot_data_utils import compute_pairwise_distance
from epyseg.deeplearning.deepl import EZDeepLearning
from batoolset.img import Img, save_as_tiff, invert
from batoolset.nps.tools import filter_nan_rows
from batoolset.ta.database.sql import query_db_and_get_results
from sql_tools import combine_single_file_queries
from batoolset.files.tools import smart_name_parser, smart_name_appender
from fishdist.wing_analysis_utils import add_to_db_sql, get_q1_q3,compute_weighted_centroid
from fishdist.stack_prediction import predict_3D_stack_from_2D_model
from timeit import default_timer as timer
from fishdist.bresenham_nd import \
    bresenham_nd
from fishdist.point_correction_and_analysis import perform_correction, \
    finalize_analysis_and_save_db
from fishdist.spot_data_utils import get_green_and_blue_dots
import os.path
import sqlite3
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from batoolset.img import Img, get_voxel_conversion_factor
from batoolset.nps.tools import convert_numpy_bbox_to_coord_pairs, get_image_bounds
from batoolset.ta.measurements.TAmeasures import distance_between_points
from batoolset.files.tools import smart_name_parser
# from personal.geom.tools import merge_coords
from fishdist.affine_transform import affine_matrix_from_points, \
    affineTransform, affine_transform_3D_image
from fishdist.spot_validation import validate_spots
from fishdist.point_pairing import pair_points_by_distance
import numpy as np
from fishdist.coordinate_utils import remove_out_of_bonds_coords_nd, get_cuboid
import pandas as pd
import seaborn as sns

# nuclear_model_to_use = 'nuclear_model_0' #'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/220916_new_nuclear_detection_model_retrained_including_gradient_fluo/nuclear_model_0.h5'
# spot_model_to_use = 'spot_model_0' #'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/221012_second_test_training_spots_with_gradients_and_imroved_ellipse_GT/spot_model_0.h5'  # se

def merge_coords(*regions, remove_dupes=True):
    """
    Merge multiple sets of coordinates into a single set.

    Args:
        *regions (np.ndarray): Variable number of regions (sets of coordinates).
        remove_dupes (bool, optional): Whether to remove duplicate coordinates. Defaults to True.

    Returns:
        np.ndarray: The merged coordinates.

    Example:
        ```python
        >>> import numpy as np
        >>> coords1 = np.array([[1, 2], [3, 4]])
        >>> coords2 = np.array([[3, 4], [5, 6]])
        >>> result = merge_coords(coords1, coords2)
        >>> print(result)
        [[1 2]
         [3 4]
         [5 6]]

        ```
    """
    merge = np.concatenate(regions, axis=0)
    if not remove_dupes:
        return merge
    else:
        return np.unique(merge, axis=0)


def apply_affine_trafo_to_image(image, affine_trafo_matrix):
    """
    Apply a 3D affine transformation to an image using the provided affine transformation matrix.

    Args:
        image (np.ndarray): 3D image array to be transformed. Expected shape is (Z, Y, X) unless your underlying transform function requires otherwise.
        affine_trafo_matrix (np.ndarray): 4×4 affine transformation matrix (or compatible format) used to transform the image.

    Returns:
        np.ndarray: The transformed image.

    Example:
        ```python
        >>> import numpy as np
        >>> image = np.random.rand(10, 100, 100)
        >>> affine = np.eye(4)
        >>> transformed = apply_affine_trafo_to_image(image, affine)
        >>> transformed.shape
        (10, 100, 100)

        ```
    """

    # Apply the 3D affine transform with spline interpolation (order=3)
    new_img = affine_transform_3D_image(
        image,
        affine_trafo_matrix,
        order=3   # interpolation order; cubic spline
    )

    return new_img





def compute_affine_transform_for_images(paths,
                                       correction_matrix_save_path,
                                       ch1=1,
                                       ch2=2,
                                       db_to_read='points_n_distances3D',
                                       USE_AFFINE_TRAFO_WITH_SHEAR=True,
                                       use_median_n_IQR_filtering_of_coords=True,
                                        # Add a parameter to indicate we don't want the 'only_in_nuclei' version
                                        ):
    """
    Compute and save a global affine transformation matrix for multiple images based on spot coordinates.

    Args:
        paths (list[str]): List of file paths to images.
        correction_matrix_save_path (str): Directory where the affine matrix and plots will be saved.
        ch1 (int, optional): First channel number. Defaults to 1.
        ch2 (int, optional): Second channel number. Defaults to 2.
        db_to_read (str, optional): Database table prefix to read spot coordinates from.
            The actual table name will be constructed as f'{db_to_read}_ch{ch1}_ch{ch2}_only_in_nuclei'.
            Defaults to 'points_n_distances3D'.
        USE_AFFINE_TRAFO_WITH_SHEAR (bool, optional): Whether to include shear in the affine transformation. Defaults to True.
        use_median_n_IQR_filtering_of_coords (bool, optional): Whether to filter coordinate pairs using median ± IQR. Defaults to True.

    Returns:
        None

    Example:
        ```python
        paths = ['sample1.tif', 'sample2.tif']
        correction_path = 'correction_results'
        compute_affine_transform_for_images(paths, correction_path, ch1=1, ch2=2)
        ```
    """


    pairs_1 = []  # more representative points may be selected
    pairs_2 = []
    full_pairs1 = []  # no selection for points
    full_pairs2 = []
    final_voxel_conversion_factor = None


    # Construct the actual table name using the channel numbers
    actual_table_name = f"{db_to_read}_ch{ch1}_ch{ch2}"

    for path in paths:
        voxel_conversion_factor = get_voxel_conversion_factor(path)

        # Ensure all datasets have same voxel size
        if final_voxel_conversion_factor is None:
            final_voxel_conversion_factor = voxel_conversion_factor
        elif not np.allclose(final_voxel_conversion_factor, voxel_conversion_factor):
            raise ValueError(
                f'Pixel size differs between samples for {path}. Chromatic aberration correction impossible.'
            )

        path_to_db = smart_name_parser(path, 'FISH.db')

        # Load all spot pairs and distances from the channel-specific table
        spot_pairs = np.asarray(query_db_and_get_results(path_to_db, f'SELECT * FROM {actual_table_name}'))
        tmp_pairs1 = spot_pairs[..., 0:3]
        tmp_pairs2 = spot_pairs[..., 3:6]
        distances = spot_pairs[..., 6]

        full_pairs1.extend(tmp_pairs1)
        full_pairs2.extend(tmp_pairs2)

        # Optional filtering based on median ± IQR
        if use_median_n_IQR_filtering_of_coords:
            q1_q3 = get_q1_q3(distances)
            coords_in_IQR = np.where((distances >= q1_q3[0]) & (distances <= q1_q3[1]))
            tmp_pairs1 = tmp_pairs1[coords_in_IQR]
            tmp_pairs2 = tmp_pairs2[coords_in_IQR]

        pairs_1.extend(tmp_pairs1)
        pairs_2.extend(tmp_pairs2)

    # --- Compute global affine transformation ---
    coords_ch0 = np.asarray(pairs_1)
    coords_ch1 = np.asarray(pairs_2)
    np.set_printoptions(suppress=True)

    M = affine_matrix_from_points(coords_ch0, coords_ch1, shear=USE_AFFINE_TRAFO_WITH_SHEAR)

    coords_ch0 = np.asarray(full_pairs1)
    coords_ch1 = np.asarray(full_pairs2)

    transformed_coords = affineTransform(coords_ch0, M)
    print('transformed_coords', transformed_coords)

    sum_error_distance = 0

    # compute orig distance and save it
    corrected_distances = []
    for iii, coord in enumerate(coords_ch0):
        sum_error_distance += distance_between_points(coord, coords_ch1[iii])
        corrected_distances.append(
            distance_between_points(coord, coords_ch1[iii], rescaling_factor=final_voxel_conversion_factor))

    corrected_distances = np.asarray(corrected_distances)
    median = np.median(corrected_distances)
    q1_q3 = get_q1_q3(corrected_distances)

    # Create channel-specific filenames for outputs
    base_filename = f"ch{ch1}_ch{ch2}"

    plot_histo(corrected_distances, median=median, q1_q3=q1_q3, title='before chromatic aberration correction')
    plt.savefig(os.path.join(correction_matrix_save_path, f'{base_filename}_pairs_before_chromatic_aberration_correction.png'))
    plt.close(fig='all')

    # --- Compute corrected distances ---
    corrected_distances = []
    for iii, coord in enumerate(transformed_coords):
        sum_error_distance += distance_between_points(coord, coords_ch1[iii])
        corrected_distances.append(
            distance_between_points(coord, coords_ch1[iii], rescaling_factor=final_voxel_conversion_factor))

    corrected_distances = np.asarray(corrected_distances)
    median = np.median(corrected_distances)
    q1_q3 = get_q1_q3(corrected_distances)
    print('median', median)
    print('q1_q3', q1_q3)

    # --- Plot distances after correction ---
    plot_histo(corrected_distances, median=median, q1_q3=q1_q3, title='acc')
    plt.savefig(os.path.join(correction_matrix_save_path, f'{base_filename}_pairs_after_acc.png'))
    plt.close(fig='all')

    print('Affine transformation matrix to apply:\n', M)
    Img(M).save(os.path.join(correction_matrix_save_path, f'{base_filename}_acc.npy'))
    Img(np.asarray(final_voxel_conversion_factor)).save(os.path.join(correction_matrix_save_path, f'voxel_size.npy')) # voxel size is independent of channels


def apply_affine_transform_to_all_images(
    paths,
    root_folder_path_to_affine_trafo_matrix,
    table_name_prefix='points_n_distances3D',
    ch1=1,
    ch2=2,
    atol=1e-6
):
    """
    Apply a precomputed affine transformation to all images and recompute corrected distances.

    This function:
        - Loads the affine transformation matrix and voxel size
        - Verifies voxel size matches the images
        - Applies the affine transformation to the first set of coordinates in a table
        - Computes distances between transformed and original second coordinates
        - Saves the corrected coordinates and distances to a new table
        - Plots the distance distribution after correction

    Args:
        paths (list[str]): List of dataset paths containing 'FISH.db'.
        root_folder_path_to_affine_trafo_matrix (str): Path where affine matrix and voxel size are saved.
            Files should be named as 'ch{ch1}_ch{ch2}_acc.npy' and
            'ch{ch1}_ch{ch2}_voxel_size.npy'.
        table_name_prefix (str, optional): Prefix of the table in each FISH.db to which the affine transformation
            will be applied. The actual table name will be constructed as f'{table_name_prefix}_ch{ch1}_ch{ch2}_only_in_nuclei'.
            Defaults to 'points_n_distances3D'.
        ch1 (int, optional): First channel number. Defaults to 1.
        ch2 (int, optional): Second channel number. Defaults to 2.
        atol (float, optional): Absolute tolerance when comparing voxel sizes. Defaults to 1e-6.

    Returns:
        None

    Example:
        ```python
        paths = ['sample1.tif', 'sample2.tif']
        root_folder = 'correction_results'
        apply_affine_transform_to_all_images(paths, root_folder, ch1=1, ch2=2)
        ```
    """
    # Construct filenames with channel information
    base_filename = f"ch{ch1}_ch{ch2}"
    path_to_affine = os.path.join(root_folder_path_to_affine_trafo_matrix, f'{base_filename}_acc.npy')

    # TODO --> maybe deactivate this !!! -> probably yes but ok for debug --> this is just for backwards compat
    if not os.path.exists(path_to_affine):
        try:
            path_to_affine = os.path.join(root_folder_path_to_affine_trafo_matrix,'acc.npy')
        except:
            path_to_affine=os.path.join(root_folder_path_to_affine_trafo_matrix,'affine_chromatic_aberration_correction.npy')


    path_to_voxel = os.path.join(root_folder_path_to_affine_trafo_matrix, 'voxel_size.npy') # there is only one voxel size file so this does not make sense

    # Construct the actual table names
    input_table_name = f"{table_name_prefix}_ch{ch1}_ch{ch2}"
    output_table_name = f"{table_name_prefix}_ch{ch1}_ch{ch2}_acc"

    np.set_printoptions(suppress=True)
    M = Img(path_to_affine)  # affine transformation matrix
    final_voxel_size = tuple(Img(path_to_voxel).tolist())  # reference voxel size

    print('Affine transformation matrix M:\n', M)
    print('Reference voxel size:', final_voxel_size)

    # Apply transformation to each dataset
    for path in paths:
        try:
            voxel_conversion_factor = get_voxel_conversion_factor(path)
            print('Processing:', path)
            print('Voxel size:', voxel_conversion_factor, 'vs reference:', final_voxel_size)

            # Skip if voxel size differs significantly
            if not np.allclose(list(voxel_conversion_factor), list(final_voxel_size), atol=atol):
                print('Voxel size mismatch: skipping affine transformation for this dataset.')
                continue

            path_to_db = smart_name_parser(path, 'FISH.db')

            # Load coordinates from database
            spot_pairs = np.asarray(
                query_db_and_get_results(path_to_db, f'SELECT * FROM {input_table_name}'))
            tmp_pairs1 = spot_pairs[..., 0:3]  # coordinates to transform
            tmp_pairs2 = spot_pairs[..., 3:6]  # reference coordinates

            # Apply affine transformation
            transformed_coords = affineTransform(tmp_pairs1, M)

            # Recompute distances using the voxel conversion factor
            distances = compute_pairwise_distance(transformed_coords, tmp_pairs2,
                                                  rescaling_factor=voxel_conversion_factor)
            distances = np.asarray(distances)[..., np.newaxis]  # add axis for stacking

            # Stack transformed coords, original coords, and distances
            total = np.hstack((transformed_coords, tmp_pairs2, distances))

            # Save corrected coordinates and distances to a new table
            add_to_db_sql(
                sql_file=path_to_db,
                table_name=output_table_name,
                headers=['z1', 'y1', 'x1', 'z2', 'y2', 'x2', 'distance'],
                data_rows=total
            )

            # Plot distance distribution after correction
            plot_distance_distribution(
                distances,
                smart_name_parser(path, f'{output_table_name}.png')
            )
        except Exception as e:
            print(f'Failed processing {path}. Error: {str(e)}')
            traceback.print_exc()


def plot_distance_distribution(distances, outputfile_name):
    """
    Compute distribution statistics for a list/array of distances, print them, and save a histogram plot to a file.

    Args:
        distances (array-like): Numeric distance values for which the distribution will be analyzed.
        outputfile_name (str): Path where the generated plot will be saved (e.g., "output.png").

    Returns:
        None

    Example:
        ```python
        distances = [1.2, 2.5, 3.0, 4.1, 2.9]
        outputfile = 'distance_histogram.png'
        plot_distance_distribution(distances, outputfile)

        ```
    """


    # Compute distribution statistics
    median = np.median(distances)
    q1_q3 = get_q1_q3(distances)

    print("median +/- IQR:", median, "+/-", q1_q3)

    # Plot histogram with annotations
    plot_histo(
        distances,
        median=median,
        q1_q3=q1_q3,
        title="after chromatic aberration correction"
    )

    # Save the figure to file
    plt.savefig(outputfile_name)

    # Close all matplotlib figures to avoid memory accumulation
    plt.close(fig='all')


def get_bounds_for_nd_coords(nd_coords):
    """
    Compute per-dimension minimum and maximum bounds for an array of N-dimensional coordinates.

    Args:
        nd_coords (array-like or np.ndarray): An array of shape (N_points, N_dims), where each row is a coordinate and each column corresponds to a dimension.

    Returns:
        np.ndarray: Array of shape (N_dims, 2), where each row contains [min, max] for the corresponding dimension.

    Example:
        ```python
        >>> import numpy as np
        >>> coords = np.array([[0, 1, 2],
        ...                    [5, -1, 3],
        ...                    [2, 4, 0]])
        >>> get_bounds_for_nd_coords(coords)
        array([[ 0,  5],
               [-1,  4],
               [ 0,  3]])


        ```
    """
    # Ensure input is a NumPy array
    if not isinstance(nd_coords, np.ndarray):
        nd_coords = np.asarray(nd_coords)

    mins = np.min(nd_coords, axis=0)
    maxs = np.max(nd_coords, axis=0)
    return np.column_stack((mins, maxs))


    # Ensure input is a NumPy array
    if not isinstance(nd_coords, np.ndarray):
        nd_coords = np.asarray(nd_coords)

    mins = np.min(nd_coords, axis=0)
    maxs = np.max(nd_coords, axis=0)
    return np.column_stack((mins, maxs))


def _create_n_dimensional_array(EXTEND_PLANAR_OR_POINT_TO_DIAMETER, nb_dimensions):
    """
    Normalize a scalar or sequence into an n-dimensional tuple.

    Args:
        EXTEND_PLANAR_OR_POINT_TO_DIAMETER (scalar, sequence, or None):
            Either a single numeric value or a sequence of values. If `None`, the function returns `None`.
            If it is a sequence of length `nb_dimensions`, the sequence is returned unchanged.
            If it is a scalar (or non-iterable), it is replicated `nb_dimensions` times.
        nb_dimensions (int): Number of dimensions that the returned tuple must match.

    Returns:
        tuple or None: A tuple of length `nb_dimensions`, or `None` if the input was `None`.

    Example:
        ```python
        >>> _create_n_dimensional_array(5, 3)
        (5, 5, 5)
        >>> _create_n_dimensional_array([1, 2, 3], 3)
        [1, 2, 3]

        ```
    """

    # If no value is provided, propagate the None
    if EXTEND_PLANAR_OR_POINT_TO_DIAMETER is None:
        return None

    # Try interpreting the input as a sequence
    try:
        if len(EXTEND_PLANAR_OR_POINT_TO_DIAMETER) == nb_dimensions:
            # Already the right length → return as-is
            return EXTEND_PLANAR_OR_POINT_TO_DIAMETER
    except:
        # Not iterable (e.g., scalar): replicate it
        dimensions = [EXTEND_PLANAR_OR_POINT_TO_DIAMETER for _ in range(nb_dimensions)]
        return tuple(dimensions)

    # Iterable but wrong length → treat as scalar and repeat
    dimensions = [EXTEND_PLANAR_OR_POINT_TO_DIAMETER for _ in range(nb_dimensions)]
    return tuple(dimensions)


def extend_ROIS_to_specified_diameter(regions, EXTEND_PLANAR_OR_POINT_TO_DIAMETER, bounds):
    """
    Extend regions of interest (ROIs) to ensure that each dimension of the region's bounding box has at least a specified diameter.

    This is useful when single points or very small planar ROIs need to be expanded to a minimum size
    (e.g., for morphological consistency or for intensity measurements in 3D volumes).

    Args:
        regions (list): List of RegionProperties objects or arrays of coordinates. Each element is either:
            - a skimage.measure.RegionProperties instance, or
            - an ndarray of shape (N_points, N_dims).
        EXTEND_PLANAR_OR_POINT_TO_DIAMETER (int, sequence, or None): Minimum desired diameter for each dimension.
            If `None`, `0`, or sums to zero, regions are returned unchanged. Passed through `_create_n_dimensional_array` to normalize shape.
        bounds (np.ndarray): Array of per-dimension [min, max] bounds used to clip final coordinates, typically from the size of the 3D image.

    Returns:
        list: A list of coordinate arrays (or RegionProperties-like data), with regions expanded if needed.

    Example:
        ```python
        >>> regions = [np.array([[2, 2, 2]])]
        >>> bounds = np.array([[0, 10], [0, 10], [0, 10]])
        >>> extend_ROIS_to_specified_diameter(regions, 3, bounds)
        [array([[1, 1, 1],
               [1, 1, 2],
               [1, 1, 3],
               [1, 2, 1],
               [1, 2, 2],
               [1, 2, 3],
               [1, 3, 1],
               [1, 3, 2],
               [1, 3, 3],
               [2, 1, 1],
               [2, 1, 2],
               [2, 1, 3],
               [2, 2, 1],
               [2, 2, 2],
               [2, 2, 3],
               [2, 3, 1],
               [2, 3, 2],
               [2, 3, 3],
               [3, 1, 1],
               [3, 1, 2],
               [3, 1, 3],
               [3, 2, 1],
               [3, 2, 2],
               [3, 2, 3],
               [3, 3, 1],
               [3, 3, 2],
               [3, 3, 3]])]

        ```
    """

    # If no extension is requested → return regions unchanged
    if (EXTEND_PLANAR_OR_POINT_TO_DIAMETER is None
        or EXTEND_PLANAR_OR_POINT_TO_DIAMETER == 0
        or np.sum(EXTEND_PLANAR_OR_POINT_TO_DIAMETER) == 0):
        return regions

    out = []

    for region in regions:

        # Extract bounding box and coordinates
        if isinstance(region, RegionProperties):
            coord_pairs = convert_numpy_bbox_to_coord_pairs(region.bbox)
            coords = region.coords
        else:  # region is assumed to be an ndarray of coordinates
            coord_pairs = get_bounds_for_nd_coords(region)
            coords = region

        # Normalize desired diameters to an n-dimensional tuple
        nd_array = _create_n_dimensional_array(
            EXTEND_PLANAR_OR_POINT_TO_DIAMETER,
            nb_dimensions=coords.shape[-1]
        )

        # Compute the size of the bounding box along each dimension
        dimension_sizes = coord_pairs[:, 1] - coord_pairs[:, 0]

        # If any dimension is smaller than desired → expansion needed
        must_be_reshaped = np.min(dimension_sizes - np.asarray(nd_array)) < 0

        if must_be_reshaped:

            # Create a coordinate “cube” of the required size.
            # NOTE: np.zeros(nd_array) is using 'nd_array' as shape, not meaning.
            zeros = np.zeros(nd_array)

            # Extract coordinates of cube (all zero-valued voxels)
            coords_cube = np.where(zeros == 0)
            coords_cube = tuple(zip(*coords_cube))
            coords_cube = np.asarray(coords_cube)

            # Compute centroid of the existing coordinates
            centroid = np.rint(np.average(coords, axis=0)).astype(int)

            # Center the cube around the centroid
            coords_cube = coords_cube + centroid - (np.asarray(nd_array) - 1) / 2.
            coords_cube = np.rint(coords_cube).astype(int)

            # Merge old and new coordinates
            coords = merge_coords(coords, coords_cube)

            # Clip coordinates to stay within imaging bounds
            region = remove_out_of_bonds_coords_nd(coords, bounds)

        out.append(region)

    return out


def find_spots_in_nuclei(spot_ROIs, nuclear_mask):
    """
    Compute the fraction of each spot ROI that overlaps with a nuclear mask.

    Each spot is represented either as a RegionProperties object or as an array of coordinates.
    The nuclear mask is a binarized image where non-zero pixels indicate nuclear regions.

    Args:
        spot_ROIs (list): List of RegionProperties or ndarray of coordinates (shape: N_points x N_dims).
        nuclear_mask (np.ndarray or None): Binary mask of nuclei. Non-zero pixels indicate nuclear regions.
            If None, all spots are considered outside nuclei.

    Returns:
        list of float: Fraction of each spot contained within the nuclear mask. Value is between 0 (no overlap) and 1 (fully inside nuclei).

    Example:
        ```python
        >>> spot_ROIs = [np.array([[0,0],[0,1]]), np.array([[5,5]])]
        >>> nuclear_mask = np.zeros((10,10), dtype=int)
        >>> nuclear_mask[0,0] = 1
        >>> find_spots_in_nuclei(spot_ROIs, nuclear_mask)
        [0.5, 0.0]


        ```
    """

    nuclear_scores = []

    # Case 1: no nuclear mask → all spots considered outside nuclei
    if nuclear_mask is None:
        nuclear_scores = [0.0 for _ in spot_ROIs]
        return nuclear_scores

    # Case 2: nuclear mask is all black or all white
    if nuclear_mask.max() == nuclear_mask.min():
        value = 1.0 if nuclear_mask.max() > 0 else 0.0
        nuclear_scores = [value for _ in spot_ROIs]
        return nuclear_scores

    # Case 3: compute fraction of each spot within nuclei
    for spot in spot_ROIs:
        # Extract coordinates if spot is a RegionProperties object
        if isinstance(spot, RegionProperties):
            spot = spot.coords

        # Extract pixel values of the nuclear mask corresponding to the spot
        pixels = nuclear_mask[tuple(spot.T)]

        # Fraction of spot pixels that are in the nucleus
        nuclear_score_of_the_spot = np.count_nonzero(pixels) / pixels.size
        nuclear_scores.append(nuclear_score_of_the_spot)

    return nuclear_scores

# can specify the min dimension of the ROI in each dimension and can extend it if too small with the value are centered --> maybe --> think about that --> shall I create a flask disk ??? --> or a square
# the minimal diameter can overcome the area criterion --> fake extend the region --> is the gaussian fit also bad in some cases because my ROI was not good ???
# CUBOID RADIUS is the dominant if it is there it defines the cube
def detect_spots(detected_spot_binary_image,
                 original_image,
                 validity_mask=None,
                 area_threshold=None,
                 minimal_diameter=None,
                 smart_detect_spots=False,
                 CUBOID_RADIUS=None,
                 gaussian_fit_3D=False,
                 voxel_size=None,
                 real_spot_size=None,
                 gaussian_fit_mode='all'):
    """
    Detect spot centroids in an image with optional filtering, ROI expansion, and sub-pixel Gaussian refinement.

    Args:
        detected_spot_binary_image (np.ndarray): Binary image where non-zero pixels indicate detected spots.
        original_image (np.ndarray): Original intensity image used for centroid computation and refinement.
        validity_mask (np.ndarray or None): Optional binary mask used to score spots based on spatial validity.
            Defaults to None.
        area_threshold (int or None): Minimum area a detected spot must have to be kept. Defaults to None.
        minimal_diameter (int or sequence or None): Minimum diameter to which planar or point ROIs are extended.
            Defaults to None.
        smart_detect_spots (bool): Whether to use a smart validation-based spot detection strategy.
            Defaults to False.
        CUBOID_RADIUS (int or sequence or None): Radius of a cuboid region used to refine centroid computation.
            Defaults to None.
        gaussian_fit_3D (bool): Whether to perform 3D Gaussian fitting for sub-pixel localization.
            Defaults to False.
        voxel_size (sequence or None): Physical voxel size used to scale Gaussian fitting.
            Defaults to None.
        real_spot_size (sequence or None): Explicit spot size used for Gaussian fitting.
            Defaults to None.
        gaussian_fit_mode (str): Mode for Gaussian fitting ("all" or selective fitting).
            Defaults to "all".

    Returns:
        np.ndarray: Array of detected spot coordinates, optionally augmented with validity scores
        and Gaussian fit success flags.

    Example:
        ```python

        ```
    """

    coords_ch0 = []

    if smart_detect_spots and CUBOID_RADIUS is None:
        rps_ch0 = validate_spots(detected_spot_binary_image, discard_spot_size_rule=[0, 1, 1],
                                 dilate_spot_size_rule=[2, 0, 0], intensity_image_for_regions_optional=original_image)
    else:
        lab_ch0 = label(detected_spot_binary_image, connectivity=None, background=0)
        rps_ch0 = regionprops(lab_ch0, intensity_image=original_image)

    if minimal_diameter is not None and CUBOID_RADIUS is None:
        print('dilating radius...')
        bounds = get_image_bounds(original_image)
        rps_ch0 = extend_ROIS_to_specified_diameter(rps_ch0, minimal_diameter,
                                                    bounds)  # could extend each dimension to increase it

    score_for_each_spot = find_spots_in_nuclei(rps_ch0, validity_mask)

    kept_regions = []
    corresponding_kept_spots_scores = []
    # pb --> ROI may have been extended before --> not a smart move --> I need a better control here
    for sss, region in enumerate(rps_ch0):
        if area_threshold is not None and region.area < area_threshold:
            # spot too small -> discard it
            continue

        if isinstance(region, RegionProperties):

            if CUBOID_RADIUS is None:
                kept_regions.append(region.coords)
                centroid = region.weighted_centroid
                coords_ch0.append(centroid)
                corresponding_kept_spots_scores.append(score_for_each_spot[sss])
            else:
                centroid = region.centroid
                cuboid_coords = get_cuboid(original_image, centroid, CUBOID_RADIUS, return_cuboid_coords_instead=True)
                coords_ch0.append(compute_weighted_centroid(cuboid_coords, original_image))
                corresponding_kept_spots_scores.append(score_for_each_spot[sss])
        else:
            if CUBOID_RADIUS is None:
                kept_regions.append(region)
                coords_ch0.append(compute_weighted_centroid(region, original_image))
                corresponding_kept_spots_scores.append(score_for_each_spot[sss])
            else:
                # get centroid and act based on that
                # not sure whether that will work due to non integer stuff --> think about that carefully
                centroid = np.average(region, axis=0)
                cuboid_coords = get_cuboid(original_image, centroid, CUBOID_RADIUS, return_cuboid_coords_instead=True)
                coords_ch0.append(compute_weighted_centroid(cuboid_coords, original_image))
                corresponding_kept_spots_scores.append(score_for_each_spot[sss])

    # --- Optional Gaussian sub-pixel refinement ---
    if gaussian_fit_3D:
        # Estimate spot radius if not provided
        if real_spot_size is None:
            average_spot = np.zeros((len(original_image.shape),))

            for sss, region in enumerate(kept_regions):
                # keep min of each dim and max of each dim
                min = np.min(region, axis=0)
                max = np.max(region, axis=0)
                cur_spot = max - min
                average_spot += cur_spot

                print('cur_spot', cur_spot)

                if gaussian_fit_mode != 'all':
                    # compute gaussian just for one spot and not for all where spot size differ

                    print('used centroid,', np.asarray([compute_weighted_centroid(region, original_image)]))
                    print('real_spot_size', tuple((cur_spot + 1).tolist()))  # MEGA TODO --> CHECK WHY +1 ????

                    coords_of_pt = get_sub_pixel_coordinates_through_gaussian_fit(
                        np.asarray([compute_weighted_centroid(region, original_image)]), original_image,
                        voxel_size=voxel_size, spot_radius=tuple((cur_spot + 1).tolist()))
                    coords_of_pt.append(corresponding_kept_spots_scores[sss])

                    coords_ch0.extend(coords_of_pt)  # MEGA TODO --> CHECK WHY +1 ????
                    print('coords_ch0', coords_ch0)

            GAUSSIAN_FIT_EXTRA_EXTENSION = 0  # just for a test --> probably remove that at some point # nb --> maybe I should keep this as asymetric still and increase by some percentage ???
            average_spot += GAUSSIAN_FIT_EXTRA_EXTENSION  # add 1 to every dimension to get real size of the spot in pixels I GUESS THE +1 WAS A MISTAKE THAT DOES NOT MAKE ANY SENSE --> RESULTS SEEM BETTER WITHOUT

            if len(kept_regions)==0: # if there is no spot --> no need to continue...
                print('error no spot kept --> check your images')
                return None

            average_spot /= len(kept_regions)
            print('average_spot diameter in pixels', average_spot)

            # BUG FIX --> get spot radius
            average_spot /= 2.
            print('average_spot radius in pixels', average_spot)

            if voxel_size is not None:
                average_spot *= np.asarray(voxel_size)

            print('average_spot radius scaled', average_spot)


        # Perform Gaussian fit on all spots
        if gaussian_fit_mode == 'all':
            print('before gaussian fit')
            old_coords = np.copy(coords_ch0)
            coords_ch0 = get_sub_pixel_coordinates_through_gaussian_fit(np.asarray(coords_ch0), original_image,
                                                                        voxel_size=voxel_size, spot_radius=tuple(
                    average_spot.tolist()))
            # check if gaussian fit was successful --> i.e. if the array has changed or not
            successfull_gaussian_fit = [np.allclose(row1, row2, atol=1e-5) for row1, row2 in
                                        zip(old_coords, coords_ch0)]
            successfull_gaussian_fit = ~np.array(successfull_gaussian_fit)
            successfull_gaussian_fit = successfull_gaussian_fit.astype(int)

            corresponding_kept_spots_scores = np.asarray(corresponding_kept_spots_scores)
            if len(corresponding_kept_spots_scores.shape) == 1:
                corresponding_kept_spots_scores = corresponding_kept_spots_scores[..., np.newaxis]

            if len(successfull_gaussian_fit.shape) == 1:
                successfull_gaussian_fit = successfull_gaussian_fit[..., np.newaxis]

            # Append scores and Gaussian fit success flag
            if validity_mask is not None:
                coords_ch0 = np.hstack((coords_ch0, corresponding_kept_spots_scores))

            if True:  # add the check for the gaussian fit
                coords_ch0 = np.hstack((coords_ch0, successfull_gaussian_fit))

            print('coords_ch0.shape', coords_ch0.shape)
            print('after gaussian fit')

    return coords_ch0


def get_sub_pixel_coordinates_through_gaussian_fit(coords, original_image, voxel_size=1, spot_radius=1):
    """
    Refine spot coordinates to sub-pixel accuracy using Gaussian fitting.

    Args:
        coords (array-like): Coordinates of detected spots with shape
            (N_spots, N_dims).
        original_image (np.ndarray): Original 2D or 3D image in which spots
            were detected.
        voxel_size (float or sequence, optional): Voxel or pixel size in each
            dimension. Defaults to 1.
        spot_radius (float or sequence, optional): Approximate radius of the
            spots used for Gaussian fitting. Defaults to 1.

    Returns:
        np.ndarray: Array of sub-pixel refined coordinates for each spot.

    Example:
        ```python
        >>> coords = np.array([[10, 15], [20, 25]])
        >>> image = np.random.random((50, 50))
        >>> refined = get_sub_pixel_coordinates_through_gaussian_fit(coords, image)
        >>> refined.shape
        (2, 2)


        ```
    """

    from bigfish.detection import fit_subpixel

    # Use BigFish's sub-pixel fitting
    spots_subpixel = fit_subpixel(
        original_image,
        coords,
        voxel_size=voxel_size,
        spot_radius=spot_radius  # consider renaming to spot_radius
    )

    return spots_subpixel



def plot_histo(distances, median=None, q1_q3=None, title=None):
    """
    Plot a histogram of distance values and annotate it with the median and interquartile range.

    Args:
        distances (array-like): List or NumPy array of numeric distance values.
        median (float, optional): Precomputed median to plot. If None, it is
            computed from `distances`. Defaults to None.
        q1_q3 (tuple[float, float], optional): Tuple containing the first and
            third quartiles (Q1, Q3). If None, they are computed from
            `distances` using `get_q1_q3`. Defaults to None.
        title (str, optional): Title to display above the histogram. Defaults
            to None.

    Returns:
        None: This function produces a plot but does not return a value.

    Example:
        ```python
        distances = [1.2, 1.5, 1.7, 2.0, 2.1]
        plot_histo(distances, title="Distance distribution")

        ```
    """

    # Handle missing data
    if distances is None:
        print("Error: empty distances file")
        return

    # Set title if given
    if title is not None:
        plt.title(title)

    # Compute median if not provided
    if median is None:
        median = np.median(distances)

    # Compute Q1/Q3 if not provided
    if q1_q3 is None:
        q1_q3 = get_q1_q3(distances)

    # Histogram plot
    plt.hist(distances)
    min_ylim, max_ylim = plt.ylim()  # Current y-axis limits

    # Plot median
    plt.axvline(
        median,
        color='k',
        linestyle='dashed',
        linewidth=2
    )
    plt.text(
        median * 1.1,
        max_ylim * 0.9,
        f"Median: {median:.3f}"
    )

    # Plot IQR boundaries
    plt.axvline(q1_q3[0], color='0.6', linestyle='dashed', linewidth=2)
    plt.axvline(q1_q3[1], color='0.6', linestyle='dashed', linewidth=2)


def finalize_quantifications_n_pairing(
    path_to_db,
    first_spot_channel=1,
    second_spot_channels=-1,
    PAIRING_THRESHOLD=250,
    voxel_conversion_factor=None,
    ONLY_DETECT_SPOTS_IN_NUCLEI=True
):
    """
    Compute spot pairing across channels and save distance statistics to a database.

    Args:
        path_to_db (str): Path to the SQLite database.
        first_spot_channel (int): Channel index for the reference spot channel.
        second_spot_channels (list[int] or None): List of channel indices for spot channels
            to pair with the first_spot_channel. If None, all channels except nuclei and
            first_spot_channel are used.
        PAIRING_THRESHOLD (float, optional): Maximum allowed pairing distance in pixels
            (unless a voxel conversion factor is provided). Defaults to 250.
        voxel_conversion_factor (tuple or list or None, optional): Optional
            (z, y, x) conversion factors used to convert pixel distances to
            physical units (e.g., micrometers). Defaults to None.
        ONLY_DETECT_SPOTS_IN_NUCLEI (bool, optional): Whether to restrict pairing
            to spots detected inside nuclei and compute penetrance statistics.
            Defaults to True.

    Returns:
        None: Results are written to the database and histogram files are saved
        to disk.

    Example:
        ```python
        path_to_db = "experiment/FISH.db"
        finalize_quantifications_n_pairing(
            path_to_db,
            first_spot_channel=1,
            second_spot_channels=[2, 3, 4],
            PAIRING_THRESHOLD=200,
            voxel_conversion_factor=(0.3, 0.1, 0.1),
            ONLY_DETECT_SPOTS_IN_NUCLEI=True
        )
        ```
    """

    PAIR_MINIMIZING_PIXEL_DISTANCE = False  # If False: use real physical distances when pairing

    # ----------------------------------------------------------------------
    # Load nuclei if required
    # ----------------------------------------------------------------------
    if ONLY_DETECT_SPOTS_IN_NUCLEI:
        nucs_ch0 = None
        try:
            query = query_db_and_get_results(path_to_db, 'SELECT * FROM nuclei')

            if query is None:
                print('No nuclei found --> ignoring pairing in nuclei')
                return

            nucs_ch0 = np.asarray(query)

        except:
            print('No nuclei found --> ignoring pairing in nuclei')

        # Load only well-scored spots for first channel
        spots_ch1 = np.asarray(
            query_db_and_get_results(path_to_db, f'SELECT * FROM spots_ch{first_spot_channel} WHERE score >= 0.5')
        )

        # Store basic penetrance estimate
        data = [[
            len(nucs_ch0),
            len(spots_ch1),
            'nb nuclei may be underestimated'
        ]]

        add_to_db_sql(
            sql_file=path_to_db,
            table_name='penetrance_estimate',
            headers=['nb_nuclei', 'nb_spots_ch1', 'warning'],
            data_rows=data
        )

    else:
        # Load unfiltered spot detections for first channel
        spots_ch1 = np.asarray(query_db_and_get_results(path_to_db, f'SELECT * FROM spots_ch{first_spot_channel}'))

    # ----------------------------------------------------------------------
    # Fix Gaussian-fit failures: remove extra columns if present
    # ----------------------------------------------------------------------
    if spots_ch1.shape[-1] == 5:
        spots_ch1 = spots_ch1[..., :-2]
    if spots_ch1.shape[-1] == 4:
        spots_ch1 = spots_ch1[..., :-1]

    # If second_spot_channels is None, get all available spot channels
    if second_spot_channels == -1:
        # Query database to find all spot channel tables
        tables = get_tables(path_to_db)
        second_spot_channels = []
        for table in tables:
            if table.startswith('spots_ch') and table != f'spots_ch{first_spot_channel}':
                try:
                    # Extract channel number more robustly
                    parts = table.split('_')
                    if len(parts) >= 2:
                        ch = int(parts[-1])
                        second_spot_channels.append(ch)
                except:
                    continue

    # Ensure second_spot_channels is a list
    if not isinstance(second_spot_channels, list):
        second_spot_channels = [second_spot_channels]

    # ----------------------------------------------------------------------
    # Process each second spot channel
    # ----------------------------------------------------------------------
    for ch2 in second_spot_channels:
        print(f"Processing pairing between channel {first_spot_channel} and channel {ch2}")

        # Load spots for the second channel
        if ONLY_DETECT_SPOTS_IN_NUCLEI:
            try:
                spots_ch2 = np.asarray(
                    query_db_and_get_results(path_to_db, f'SELECT * FROM spots_ch{ch2} WHERE score >= 0.5')
                )
            except:
                continue
        else:
            try:
                spots_ch2 = np.asarray(query_db_and_get_results(path_to_db, f'SELECT * FROM spots_ch{ch2}'))
            except:
                continue

        if spots_ch2 is None: # if the channel associated table does not exist -> just ignore and move on
            continue

        if not spots_ch2.shape:
            continue

        # Fix Gaussian-fit failures: remove extra columns if present
        if spots_ch2.shape[-1] == 5:
            spots_ch2 = spots_ch2[..., :-2]
        if spots_ch2.shape[-1] == 4:
            spots_ch2 = spots_ch2[..., :-1]

        # ----------------------------------------------------------------------
        # Pairing step
        # ----------------------------------------------------------------------
        # Convert threshold to µm if applicable
        if not PAIR_MINIMIZING_PIXEL_DISTANCE:
            threshold = (
                PAIRING_THRESHOLD
                if voxel_conversion_factor is None
                else PAIRING_THRESHOLD * voxel_conversion_factor[1]
            )

            paired = pair_points_by_distance(
                spots_ch1,
                spots_ch2,
                threshold=threshold,
                rescaling_factor=voxel_conversion_factor
            )
        else:
            paired = pair_points_by_distance(
                spots_ch1,
                spots_ch2,
                threshold=PAIRING_THRESHOLD
            )

        try:
            # ----------------------------------------------------------------------
            # Compute pairwise distances
            # ----------------------------------------------------------------------
            if not PAIR_MINIMIZING_PIXEL_DISTANCE:
                distances = compute_pairwise_distance(
                    list(paired.keys()),
                    list(paired.values()),
                    rescaling_factor=None
                )
            else:
                distances = compute_pairwise_distance(
                    list(paired.keys()),
                    list(paired.values()),
                    rescaling_factor=voxel_conversion_factor
                )

            median = np.median(distances)
            q1_q3 = get_q1_q3(distances)

            pts1 = np.asarray(list(paired.keys()))
            pts2 = np.asarray(list(paired.values()))
            distances = np.asarray(distances)[..., np.newaxis]

            # Normalize to µm if needed
            if voxel_conversion_factor is not None and not PAIR_MINIMIZING_PIXEL_DISTANCE:
                pts1 /= voxel_conversion_factor
                pts2 /= voxel_conversion_factor

            # Final output array for DB insertion
            total = np.hstack((pts1, pts2, distances))

            # Generate table name with channel numbers in the new format
            if ONLY_DETECT_SPOTS_IN_NUCLEI:
                table_name = f'points_n_distances3D_only_in_nuclei_ch{first_spot_channel}_ch{ch2}'
            else:
                table_name = f'points_n_distances3D_ch{first_spot_channel}_ch{ch2}'

            add_to_db_sql(
                sql_file=path_to_db,
                table_name=table_name,
                headers=[
                    'z1', 'y1', 'x1',
                    'z2', 'y2', 'x2',
                    'distance'
                ],
                data_rows=total
            )

            # Average pairing displacement vector
            average_displacement = np.average(
                np.asarray(list(paired.keys())) - np.asarray(list(paired.values())),
                axis=0
            )
            print(f'Average displacement between ch{first_spot_channel} and ch{ch2}:', average_displacement)

            # Histogram drawing
            unit = 'pixels' if voxel_conversion_factor is None else 'µm'
            plot_histo(
                distances,
                median=median,
                q1_q3=q1_q3,
                title=f'Pairing distance between ch{first_spot_channel} and ch{ch2} in {unit} (threshold: ' +
                      str(PAIRING_THRESHOLD if voxel_conversion_factor is None
                          else PAIRING_THRESHOLD * voxel_conversion_factor[1]) + ')'
            )

            # Save histogram with channel numbers in filename
            fig_name = f'pairs_distances_ch{first_spot_channel}_ch{ch2}'
            if ONLY_DETECT_SPOTS_IN_NUCLEI:
                fig_name = f'pairs_distances_only_in_nuclei_ch{first_spot_channel}_ch{ch2}'

            plt.savefig(
                smart_name_parser(
                    smart_name_parser(path_to_db, 'parent'),
                    fig_name + '.png'
                )
            )
            plt.close(fig='all')

        except Exception as e:
            traceback.print_exc()
            print(f'Error in measuring distances for channels {first_spot_channel} and {ch2}: {str(e)}')

def plot_all_paired_spots(paths, ch1=1, ch2=2):
    """
    Render paired spot coordinates as 3D lines directly on the original image for all
    available tables that start with 'points_n_distances3D'.

    Args:
        paths (list[str]): Paths to images from which the associated FISH
            databases are inferred.

    Returns:
        None: This function generates and saves binary images with rendered
        3D lines for each table but does not return a value.
    """
    for path in paths:
        try:
            # Locate the database corresponding to the image path
            path_to_db = smart_name_parser(path, 'FISH.db')

            # Get all tables in the database that match the pattern
            # all_tables = get_tables(path_to_db)
            # pair_tables = [table for table in all_tables
            #               if table.startswith('points_n_distances3D')]
            #
            # print(f"Found {len(pair_tables)} tables matching 'points_n_distances3D' pattern")

            # Process each pair table
            # for table in pair_tables:

            table = f'points_n_distances3D_only_in_nuclei_ch{ch1}_ch{ch2}'
            try:
                print(f"Processing table: {table}")

                # Only 6 columns are needed: z1,y1,x1,z2,y2,x2
                spot_pairs = np.asarray(
                    query_db_and_get_results(path_to_db, f'SELECT * FROM {table}')
                )

                # Skip if no pairs found
                if spot_pairs.size == 0:
                    print(f"No pairs found in table {table}, skipping")
                    continue

                # Split table into channel 1 and channel 2 coordinates
                tmp_pairs1 = spot_pairs[..., 0:3]
                tmp_pairs2 = spot_pairs[..., 3:6]

                # Load corresponding image to obtain shape
                image_for_drawing = Img(path)
                image_for_drawing = np.zeros(
                    shape=image_for_drawing.shape[0:3],
                    dtype=np.uint8
                )

                # Draw the 3D Bresenham line for each pair
                for pt1, pt2 in zip(tmp_pairs1, tmp_pairs2):
                    coords = bresenham_nd(pt1, pt2)  # list of voxels forming the line
                    bounds = get_image_bounds(image_for_drawing)
                    coords_in_bounds_only = remove_out_of_bonds_coords_nd(coords, bounds)

                    # Paint the line in white (255)
                    image_for_drawing[tuple(coords_in_bounds_only.T)] = 255

                # Generate filename based on table name
                output_filename = smart_name_parser(path, f'pairs_ch{ch1}_ch{ch2}.tif')

                # Save result
                Img(image_for_drawing, dimensions='hwc').save(
                    output_filename,
                    mode='raw'
                )

                print(f"Saved paired spots visualization to: {output_filename}")

            except Exception as e:
                print(f'Error processing table {table}: {str(e)}')
                traceback.print_exc()

        except Exception as e:
            print(f'Error processing path {path}: {str(e)}')
            traceback.print_exc()


def get_spot_channels(image_shape, ch_nuclei, first_spot_channel, second_spot_channel=None):
    """
    Returns a list of spot channel indices to use for image analysis.

    If `second_spot_channel` is None, all channels except `ch_nuclei` and `first_spot_channel` are included.
    If `second_spot_channel` is an integer, only that channel is included.
    If `second_spot_channel` is a list, that list is returned directly.

    Args:
        image_shape (tuple): Shape of the image as (z, h, w, c).
        ch_nuclei (int): Index of the nucleus channel.
        first_spot_channel (int): Index of the first spot channel.
        second_spot_channel (int or list or None, optional):
            Index of the second spot channel. If None, all channels except `ch_nuclei` and
            `first_spot_channel` are used. If an integer, only that channel is used.
            If a list, that list is returned directly. Defaults to None.

    Returns:
        list: List of channel indices to use for spot analysis.

    Example:
        >>> image_shape = (10, 256, 256, 5)  # (z, h, w, c)
        >>> ch_nuclei = 0
        >>> first_spot_channel = 1
        >>> second_spot_channel = None
        >>> get_spot_channels(image_shape, ch_nuclei, first_spot_channel, second_spot_channel)
        [2, 3, 4]

        >>> second_spot_channel = 3
        >>> get_spot_channels(image_shape, ch_nuclei, first_spot_channel, second_spot_channel)
        [3]

        >>> second_spot_channel = [2, 4]
        >>> get_spot_channels(image_shape, ch_nuclei, first_spot_channel, second_spot_channel)
        [2, 4]

    """
    if second_spot_channel is None:
        z, h, w, c = image_shape
        all_channels = list(range(c))
        # Remove nucleus and first spot channel
        spot_channels = [ch for ch in all_channels if ch != ch_nuclei and ch != first_spot_channel]
    else:
        if isinstance(second_spot_channel, int):
            spot_channels = [second_spot_channel]
        else:
            spot_channels = second_spot_channel

    return spot_channels

def negative_to_positive_index(arr_shape, index):
    """
    Converts a negative index to a positive index for a NumPy ndarray.

    Args:
        arr_shape (tuple): Shape of the NumPy array (e.g., (z, h, w, c)).
        index (int): The index to convert (can be negative or positive).

    Returns:
        int: The positive index.

    Raises:
        IndexError: If the index is out of bounds for the given array shape.

    Example:
        >>> arr_shape = (10, 256, 256, 5)
        >>> negative_to_positive_index(arr_shape, -1)
        4
        >>> negative_to_positive_index(arr_shape, 2)
        2
    """
    try:
        index = int(index)
    except:
        pass
    if isinstance(index, int):
        if index < 0:
            # Convert negative index to positive
            return index + arr_shape[-1] if len(arr_shape) > 0 else index
        else:
            # Ensure the positive index is within bounds
            if index >= arr_shape[-1]:
                # raise IndexError(f"Index {index} is out of bounds for axis with size {arr_shape[-1]}")
                return None
            return index
    else:
        # print('index', index, type(index))
        raise TypeError("Index must be an integer")

def segment_spots_and_nuclei(
    paths,
    ch_nuclei=0,
    first_spot_channel=1,
    second_spot_channel=-1,
    __autoskip=True,
    channels_to_blur=None,
    blur_mode='recursive2D',
    deep_learning='rapid',
    nuclear_model_to_use = 'nuclear_model_0',
    spot_model_to_use = 'spot_model_0'
):
    """
    Segment nuclei and spot channels in microscopy images using deep-learning models.

    Args:
        paths (list[str]): List of paths to input images. Each image is expected
            to be a 4D stack with dimensions (z, y, x, c).
        ch_nuclei (int or None, optional): Index of the nuclear channel. If None,
            nuclear segmentation is skipped. Defaults to 0.
        first_spot_channel (int, optional): Channel index for the first spot
            signal. Defaults to 1.
        second_spot_channel (int, optional): Channel index for the second spot
            signal. Defaults to -1 (last channel).
        __autoskip (bool, optional): Whether to skip segmentation if the output
            files already exist. Defaults to True.
        channels_to_blur (list[int] or None, optional): List of channel indices
            to blur before segmentation. Defaults to None.
        blur_mode (str, optional): Blurring mode passed to `blur_3D`.
            Defaults to "recursive2D".
        deep_learning (str, optional): Flag controlling prediction behavior.
            If it contains "rapid" or "fast", slow averaging augmentations are
            disabled. Defaults to "rapid".

    Returns:
        None: Segmentation results are written to disk as TIFF files.

    Example:
        ```python
        paths = ["img1.tif", "img2.tif"]
        segment_spots_and_nuclei(
            paths,
            ch_nuclei=0,
            first_spot_channel=1,
            second_spot_channel=2,
            channels_to_blur=[0, 1],
            blur_mode="recursive2D"
        )

        ```
    """

    USE_STACK_NORMALIZATION = True

    for path in paths:
        start2 = timer()

        img = Img(path)
        print('processing img:', path)

        # Must be a 4D array: z, y, x, c
        if len(img.shape) < 4:
            print('ERROR: the image is not a stack --> skipping...')
            continue

        print('img.shape', img.shape)

        # -------------------------------------------------------------
        # Optional: blur selected channels to reduce noise
        # -------------------------------------------------------------
        if channels_to_blur:
            for ch in channels_to_blur:
                ch = negative_to_positive_index(img.shape,ch)
                if ch is None: # out of bonds index --> skip
                    continue
                print(f'blurring channel {ch} using {blur_mode}')
                img[..., ch] = blur_3D(img[..., ch], mode=blur_mode)

        # -------------------------------------------------------------
        # Extract channels
        # -------------------------------------------------------------
        # try:
        #     nuc = None if ch_nuclei is None else img[..., ch_nuclei]
        # except Exception:
        #     nuc = None
        #     print('nuclear channel undefined --> assuming no nuclei')
        #
        # spot_ch1 = img[..., first_spot_channel]
        # # NOTE: dirty hack mentioned in original code
        # spot_ch2 = img[..., second_spot_channel]
        #
        # # free memory early
        # del img

        if second_spot_channel is None:
            spot_channels = get_spot_channels(img.shape, ch_nuclei, first_spot_channel, second_spot_channel)
            second_spot_channel = spot_channels
        else:
            if not isinstance(second_spot_channel, list):
                second_spot_channel = [second_spot_channel]
            spot_channels = second_spot_channel # in theory I should check that there is no spot channel in there, I hope the user will not make mistakes

        # import sys
        # print('second_spot_channel',second_spot_channel)
        # sys.exit(0)

        all_channels_to_analyze = [ch_nuclei, first_spot_channel]+spot_channels

        all_channels_to_analyze = [negative_to_positive_index(img.shape, ch) for ch in all_channels_to_analyze] # fix indices

        second_spot_channel= [negative_to_positive_index(img.shape, ch) for ch in second_spot_channel] # fix indices

        # -------------------------------------------------------------
        # Deep learning segmentation for each channel
        # Channels in order: [nucleus, spot1, spot2]
        # -------------------------------------------------------------
        for iii, ch in enumerate(all_channels_to_analyze):
            if ch is None: # channel out of bonds --> ignore
                continue

            channel_img = img[..., ch]

            output_name = smart_name_parser(path, f'ch{ch}.tif')

            # Skip segmentation if output exists
            if __autoskip and os.path.exists(output_name):
                print(f'skipping {path} channel {ch} (already exists)')
                continue

            if channel_img is None:
                continue  # nothing to segment

            # ---------------------------------------------------------
            # Prediction parameters
            # ---------------------------------------------------------
            TILE_WIDTH = 256
            TILE_HEIGHT = 256
            TILE_OVERLAP = 32

            predict_parameters = {
                "input_channel_of_interest": None,
                "default_input_tile_width": TILE_WIDTH,
                "default_input_tile_height": TILE_HEIGHT,
                "default_output_tile_width": TILE_WIDTH,
                "default_output_tile_height": TILE_HEIGHT,
                "tile_width_overlap": TILE_OVERLAP,
                "tile_height_overlap": TILE_OVERLAP,
                "hq_predictions": "mean",
                "hq_pred_options": "Only use pixel preserving augs (Recommended for CARE-like models/surface extraction)",
                "post_process_algorithm": None,
                "input_normalization": {
                    "method": "Rescaling (min-max normalization)",
                    "range": [0, 1],
                    "individual_channels": True
                }
            }

            # Rapid mode → disable averaging/augmentations
            if 'apid' in deep_learning or 'ast' in deep_learning:
                print('rapid deep learning segmentation ON')
                predict_parameters["hq_predictions"] = None

            # ---------------------------------------------------------
            # Load model (spot or nucleus)
            # ---------------------------------------------------------
            deepTA = EZDeepLearning()

            # try a local hack to avoid having to change the archived deepTA code!
            new_models = {
                'spot_model_0': {
                    'url': 'https://gitlab.com/baigouy/models/raw/master/spot_model_0.h5',
                    'input_dims': '2D',
                    'md5': '00348f5260ee4542768c8ec5ef4dcc1d',
                    'architecture': 'Linknet',
                    'backbone': 'vgg16',
                    'activation': 'sigmoid',
                    'classes': 1,
                    'input_width': None,
                    'input_height': None,
                    'input_channels': 1,
                    'version': 1,
                    'model': 'spot_model_0',
                },
                'nuclear_model_0': {
                    'url': 'https://gitlab.com/baigouy/models/raw/master/nuclear_model_0.h5',
                    'input_dims': '2D',
                    'md5': '1cf4d006b8b64990d6069561f1aa43ce',
                    'architecture': 'Linknet',
                    'backbone': 'vgg16',
                    'activation': 'sigmoid',
                    'classes': 1,
                    'input_width': None,
                    'input_height': None,
                    'input_channels': 1,
                    'version': 1,
                    'model': 'nuclear_model_0',
                }
            }
            deepTA.pretrained_models.update(new_models)

            # DO SPECIFY CHANNEL OF INTEREST IF MULTI CHANNELS
            predict_parameters["input_channel_of_interest"] = None

            if iii != 0:
                if spot_model_to_use in ('nuclear_model_0', 'spot_model_0'):
                    deepTA.load_or_build(architecture='Linknet', backbone='vgg16', activation='sigmoid', classes=1,
                                         pretraining=spot_model_to_use)  # --> ok that works and since I don't want to bother I'll use that
                else:
                    deepTA.load_or_build(model=spot_model_to_use)
            else:
                if nuclear_model_to_use in ('nuclear_model_0', 'spot_model_0'):
                    deepTA.load_or_build(architecture='Linknet', backbone='vgg16', activation='sigmoid', classes=1,
                                         pretraining=nuclear_model_to_use)  # --> ok that works and since I don't want to bother I'll use that
                else:
                    deepTA.load_or_build(model=nuclear_model_to_use)

            # ---------------------------------------------------------
            # Predict full stack
            # ---------------------------------------------------------
            SPLIT_ALL = False
            if SPLIT_ALL:
                # split into chunks along Y-axis to avoid OOM
                splits = np.array_split(channel_img, 4, axis=1)
                del channel_img

                for idx, s in enumerate(splits):
                    splits[idx] = predict_3D_stack_from_2D_model(
                        deepTA,
                        s,
                        apply_normalization_to_entire_stack_before=USE_STACK_NORMALIZATION,
                        **predict_parameters
                    )

                prediction = np.concatenate(splits, axis=1)

            else:
                prediction = predict_3D_stack_from_2D_model(
                    deepTA,
                    channel_img,
                    apply_normalization_to_entire_stack_before=USE_STACK_NORMALIZATION,
                    **predict_parameters
                )

            # Save only first channel
            save_as_tiff(prediction[..., np.newaxis], output_name)

        print('time for image:', path, timer() - start2)


def set_nb_channels(db_path, nb_channels):
    """
    Inserts the number of channels of an image into an SQLite database table named 'nb_channels'
    with a single column 'nb'.

    Args:
        db_path (str): The path to the SQLite database file.
        nb_channels (int): The number of channels in the image.

    Returns:
        None.
    """

    # Delete table if exists
    drop_table_if_exists(db_path, 'nb_channels', verbose=False)

    if nb_channels is None:
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    c = conn.cursor()

    # Create the 'nb_channels' table
    c.execute('CREATE TABLE nb_channels ("nb_channels")')

    # Insert the number of channels into the table
    c.execute('INSERT INTO nb_channels (nb_channels) VALUES (?)', (nb_channels,))

    # Commit the transaction
    conn.commit()

    # Close the cursor and database connection
    c.close()
    conn.close()

def get_nb_channels(db_path):
    """
    Retrieves the number of channels from an SQLite database file.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        int or None: The number of channels if found, None otherwise.

    Examples:
        ```python
        If the 'nb_channels' table in the database contains the value 3, then:

        get_nb_channels('example.db')
        3

        If the 'nb_channels' table does not exist in the database file:

        get_nb_channels('example.db')
        None

        If the database file does not exist:

        get_nb_channels('nonexistent.db')
        None
        ```

    """
    if not os.path.exists(db_path):
        return None

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    c = conn.cursor()

    # Select the row from the 'nb_channels' table
    try:
        c.execute('SELECT nb_channels FROM nb_channels')
        row = c.fetchone()

        # If a row was found, return the number of channels
        if row is not None:
            nb_channels = row[0]
        else:
            nb_channels = None
    except:
        # Most likely table does not exist --> return None
        nb_channels = None

    # Close the cursor and database connection
    c.close()
    conn.close()

    return nb_channels



def detect_spots_and_nuclei(
    paths,
    ch_nuclei=0,
    first_spot_channel=1,
    second_spot_channel=-1,
    area_threshold=None,
    channels_to_blur=None,
    blur_mode='recursive2D',
    threshold_nuclei=0.5,
    threshold_spot=0.5,
):
    """
    Detect 3D nuclei and spot coordinates from deep-learning probability maps.

    Args:
        paths (list[str]): List of input image paths.
        ch_nuclei (int or None, optional): Channel index of nuclei in the original
            image. If None, nuclei detection on the raw image is skipped.
            Defaults to 0.
        spot_channels (list[int], optional): List of channel indices for spot detection.
            Defaults to [1, 2].
        area_threshold (int or None, optional): Minimum ROI area threshold passed
            to `detect_spots`. Defaults to None.
        channels_to_blur (list[int] or None, optional): Channels in the raw image
            to blur before detection. Defaults to None.
        blur_mode (str, optional): Blur mode for 3D smoothing. Defaults to
            "recursive2D".
        threshold_nuclei (float, optional): Binarization threshold for the
            deep-learning probability map of the nuclear channel. Defaults to 0.5.
        threshold_spot (float, optional): Binarization threshold for the
            deep-learning probability map of spot channels. Defaults to 0.5.

    Returns:
        None: Detected nuclei and spot coordinates are written to the database.

    Example:
        ```python
        paths = ["img1.tif", "img2.tif"]
        detect_spots_and_nuclei(
            paths,
            ch_nuclei=0,
            spot_channels=[1, 2, 3, 4],
            area_threshold=10,
            channels_to_blur=[0, 1],
            blur_mode="recursive2D",
            threshold_nuclei=0.5,
            threshold_spot=0.6
        )
        ```
    """

    SMART_FILTER_AND_EXPAND_ROIs = False   # For advanced morphology filtering (currently disabled)
    COMPUTE_REAL_DISTANCE = True           # Determines whether voxel size is extracted
    EXTEND_PLANAR_OR_POINT_TO_DIAMETER = None
    CUBOID_RADIUS = (2, 3, 3)              # None disables cuboid extraction

    if SMART_FILTER_AND_EXPAND_ROIs:
        CUBOID_RADIUS = None

    # Gaussian fit mode overrides above settings
    GAUSSIAN_FIT_3D = True
    if GAUSSIAN_FIT_3D:
        CUBOID_RADIUS = None
        EXTEND_PLANAR_OR_POINT_TO_DIAMETER = None
        SMART_FILTER_AND_EXPAND_ROIs = True

    for path in paths:
        print('path --> ', path)
        orig = Img(path)

        spot_channels = get_spot_channels(orig.shape, ch_nuclei, first_spot_channel, second_spot_channel)
        spot_channels = [first_spot_channel] + spot_channels

        # ---------------------------------------------------------
        # Optional: blur certain channels in raw image
        # ---------------------------------------------------------
        if channels_to_blur:
            for ch in channels_to_blur:
                print(f"blurring channel {ch} using {blur_mode}")
                orig[..., ch] = blur_3D(orig[..., ch], mode=blur_mode)

        # ---------------------------------------------------------
        # Get voxel size (needed for Gaussian fit with real units)
        # ---------------------------------------------------------
        voxel_conversion_factor = None
        if COMPUTE_REAL_DISTANCE:
            voxel_conversion_factor = get_voxel_conversion_factor(orig, return_111_if_none=False)

        # ---------------------------------------------------------
        # Get nb of channels in raw image
        # ---------------------------------------------------------
        set_nb_channels(smart_name_parser(path, 'TA') + '/FISH.db', orig.shape[-1])

        if voxel_conversion_factor is not None:
            db = smart_name_parser(path, 'TA') + '/FISH.db'
            set_voxel_size(db, voxel_conversion_factor)

        if GAUSSIAN_FIT_3D and voxel_conversion_factor is None:
            print("voxel size not found --> assuming isotropic voxels")
            voxel_conversion_factor = np.asarray([1., 1., 1.])

        # ---------------------------------------------------------
        # Extract original raw channels (Z,Y,X per channel)
        # ---------------------------------------------------------
        orig_nucs = None if ch_nuclei is None else orig[..., ch_nuclei]

        # ---------------------------------------------------------
        # Load deep-learning probability maps for nuclei
        # ---------------------------------------------------------
        try:
            nucs_ch0 = Img(smart_name_parser(path, f'ch{ch_nuclei}.tif')) > threshold_nuclei
        except Exception:
            nucs_ch0 = None
            print("nuclear DL channel not found --> ignoring")

        # ---------------------------------------------------------
        # Detect nuclei coordinates (simplified detection)
        # ---------------------------------------------------------
        if nucs_ch0 is not None:
            nucs_ch0 = np.asarray(detect_spots(nucs_ch0, orig_nucs))
            add_to_db_sql(
                sql_file=smart_name_parser(path, 'FISH.db'),
                table_name='nuclei',
                headers=['z', 'y', 'x'],
                data_rows=nucs_ch0
            )

        # ---------------------------------------------------------
        # Process each spot channel dynamically
        # ---------------------------------------------------------
        for i, ch in enumerate(spot_channels):
            print(f"Processing spot channel {ch}")

            ch = negative_to_positive_index(orig.shape, ch)
            if ch is None:  # channel out of bonds --> ignore
                continue

            # Extract raw channel
            orig_ch = orig[..., ch]

            try:
                nucs_ch0 = Img(smart_name_parser(path, f'ch{ch_nuclei}.tif')) > threshold_nuclei
            except Exception:
                nucs_ch0 = None

            # Load deep-learning probability map for spot channel
            try:
                spots_ch = Img(smart_name_parser(path, f'ch{ch}.tif')) > threshold_spot
            except Exception:
                print(f"spot DL channel {ch} not found --> skipping")
                continue

            # print('TESTING', spots_ch.shape, orig_ch.shape, nucs_ch0.shape) # indeed the last one is not ok --> why is that I guess I must mutate it somewhere

            # Run spot detection
            spots_ch = np.asarray(
                detect_spots(
                    spots_ch,
                    orig_ch,
                    validity_mask=nucs_ch0,
                    area_threshold=area_threshold,
                    minimal_diameter=EXTEND_PLANAR_OR_POINT_TO_DIAMETER,
                    smart_detect_spots=SMART_FILTER_AND_EXPAND_ROIs,
                    CUBOID_RADIUS=CUBOID_RADIUS,
                    gaussian_fit_3D=GAUSSIAN_FIT_3D,
                    voxel_size=voxel_conversion_factor
                )
            )

            # Remove invalid entries
            spots_ch = filter_nan_rows(spots_ch)

            # Decide table headers based on available columns
            header = ['z', 'y', 'x']
            if spots_ch.shape[-1] == 5:
                header = ['z', 'y', 'x', 'score', 'successful_gaussian_fit']
            elif spots_ch.shape[-1] == 4:
                header = ['z', 'y', 'x', 'score']

            # Save spot detections to DB
            add_to_db_sql(
                sql_file=smart_name_parser(path, 'FISH.db'),
                table_name=f'spots_ch{ch}',
                headers=header,
                data_rows=spots_ch
            )


def pair_spots(paths, PAIRING_THRESHOLD=250, first_spot_channel=1, second_spot_channels=-1):
    """
    Align and pair detected spots across channels for multiple datasets, both
    with and without restricting detection to nuclear regions.

    For each dataset path, the pairing pipeline is executed twice:
    once considering all detected spots, and once considering only spots
    located inside nuclei.

    Args:
        paths (list[str]): List of file paths or dataset identifiers to process.
        PAIRING_THRESHOLD (float, optional): Maximum distance (in voxel units or
            scaled units) allowed for pairing spots. Defaults to 250.
        first_spot_channel (int, optional): Channel index for the reference spot channel.
            Defaults to 1.
        second_spot_channels (list[int] or None, optional): List of channel indices for spot channels
            to pair with the first_spot_channel. If None, all channels except nuclei and
            first_spot_channel are used. Defaults to None.

    Returns:
        None: Results are written to the corresponding databases.

    Example:
        ```python
        paths = ["sample_01.tif", "sample_02.tif"]
        pair_spots(
            paths,
            PAIRING_THRESHOLD=200,
            first_spot_channel=1,
            second_spot_channels=[2, 3, 4]
        )
        ```
    """

    for path in paths:
        # Load voxel scaling factor for this dataset
        voxel_conversion_factor = get_voxel_conversion_factor(
            path,
            return_111_if_none=False
        )

        # Run spot quantification/pairing twice:
        # first considering all spots, then restricting to nuclear spots
        for value in [False, True]:
            finalize_quantifications_n_pairing(
                smart_name_parser(path, 'FISH.db'),
                first_spot_channel=first_spot_channel,
                second_spot_channels=second_spot_channels,
                ONLY_DETECT_SPOTS_IN_NUCLEI=value,
                voxel_conversion_factor=voxel_conversion_factor,
                PAIRING_THRESHOLD=PAIRING_THRESHOLD
            )


def do_controls_for_easy_check(paths, ch1=1, ch2=2, ch_nuclei=0):
    """
    Generate quick visual control images for multi-channel datasets.

    For each dataset path, the function loads specified channels, processes
    them (optional inversion, normalization, dilation), computes merged 3D
    stacks and maximum projections, and saves reduced-size versions for
    visual inspection.

    Args:
        paths (list[str]): List of dataset paths to process.
        ch1 (int, optional): First channel index. Defaults to 1.
        ch2 (int, optional): Second channel index. Defaults to 2.
        ch_nuclei (int, optional): Nuclei channel index. Defaults to 0.

    Returns:
        None: The function saves control images to disk for each dataset.

    Example:
        ```python
        paths = ["sample_01.tif", "sample_02.tif"]
        do_controls_for_easy_check(paths, ch1=1, ch2=2, ch_nuclei=0)
        ```
    """

    for path in paths:
        nuclei_channel = None
        print(f"Processing {path} with channels: nuclei={ch_nuclei}, ch1={ch1}, ch2={ch2}")

        # Load nuclei channel if file exists
        if os.path.exists(smart_name_parser(path, f'ch{ch_nuclei}.tif')):
            nuclei_channel = Img(smart_name_parser(path, f'ch{ch_nuclei}.tif'))
            max_nuclei = np.max(nuclei_channel, axis=0)
            max_nuclei = invert(max_nuclei)

        if nuclei_channel is not None:
            nuclei_channel = invert(nuclei_channel)

        # Load specified channels
        try:
            ch1_img = Img(smart_name_parser(path, f'ch{ch1}.tif'))
        except:
            print(f"Channel {ch1} not found for {path}")
            continue

        try:
            ch2_img = Img(smart_name_parser(path, f'ch{ch2}.tif'))
        except:
            print(f"Channel {ch2} not found for {path}")
            continue

        # Process pairs channel if present
        if os.path.exists(smart_name_parser(path, f'pairs_ch{ch1}_ch{ch2}.tif')):
            ch3 = Img(smart_name_parser(path, f'pairs_ch{ch1}_ch{ch2}.tif')).astype(float)
            ch3 = ch3 / ch3.max()  # normalize to 0-1

            # Maximum projection along the first axis
            max_connection = np.max(ch3, axis=0)
            max_connection = ((max_connection > 0.5) * 255).astype(np.uint8)

            # Dilate to emphasize spots/connections
            s = generate_binary_structure(2, 1)
            max_connection = grey_dilation(max_connection, footprint=s)
            max_connection = grey_dilation(max_connection, footprint=s)

            # Save dilated max projection
            Img(max_connection, dimensions='hw').save(
                smart_name_parser(path, f'control_max_proj_pairs_only_dilated_ch{ch1}_ch{ch2}.tif')
            )
            del max_connection

            # Stack channels for 3D merged image
            if nuclei_channel is None:
                merge = np.stack((ch1_img, ch2_img, ch3), axis=-1)
            else:
                merge = np.stack((nuclei_channel, ch1_img, ch2_img, ch3), axis=-1)
            del ch1_img, ch2_img, ch3

        else:
            # Stack only available channels
            if nuclei_channel is None:
                merge = np.stack((ch1_img, ch2_img), axis=-1)
            else:
                merge = np.stack((nuclei_channel, ch1_img, ch2_img), axis=-1)
            del ch1_img, ch2_img

        # Compute max projection and threshold for reduced-size visualization
        max_proj = np.max(merge, axis=0)
        tmp = (merge > 0.5).astype(np.uint8)
        del merge
        merge = tmp * 255
        Img(merge, dimensions='dhwc').save(smart_name_parser(path, f'control_3D_ch{ch1}_ch{ch2}.tif'))
        del merge

        # Replace nuclei channel in max projection with original if present
        if nuclei_channel is not None:
            max_proj[..., 0] = max_nuclei
            del nuclei_channel

        # Threshold and save final max projection
        max_proj = ((max_proj > 0.5) * 255).astype(np.uint8)
        Img(max_proj, dimensions='hwc').save(
            smart_name_parser(path, f'control_max_proj_ch{ch1}_ch{ch2}.tif')
        )
        del max_proj



def run_analysis(
    paths,
    correction_matrix_save_path,
    PAIRING_THRESHOLD=60,
    ch_nuclei=0,
    first_spot_channel=1,
    second_spot_channel=-1,
    area_threshold=None,
    channels_to_blur=None,
    blur_mode='recursive2D',
    deep_learning='rapid',
    threshold_spot_ch1=0.5,
    threshold_nuclei=0.5,
    threshold_spot_ch2=0.5,
    RUN_SEG=True,
    RUN_REG=False,
    RUN_DISTANCE_MEASUREMENTS=True,
    RUN_CTRLS=True,
    RUN_GAUSSIAN_FIT=True,
    nuclear_model_to_use = 'nuclear_model_0',
    spot_model_to_use = 'spot_model_0',
    list_pairs_for_reg=None
):
    """
    Execute a full pipeline for FISH spot and nuclei analysis including segmentation,
    spot detection, pairing, registration, distance measurements, controls, and plotting.

    Args:
        paths (list[str]): List of dataset paths to analyze.
        correction_matrix_save_path (str): Path to save affine transformation / registration matrices.
        PAIRING_THRESHOLD (float, optional): Maximum distance for pairing spots (default 60).
        ch_nuclei (int, optional): Channel index for nuclei (default 0).
        first_spot_channel (int, optional): Channel index for first spot type (default 1).
        second_spot_channel (int, optional): Channel index for second spot type (default -1).
        area_threshold (float or None, optional): Minimum area threshold for detected nuclei.
        channels_to_blur (list[int] or None, optional): Channels to apply blur to during preprocessing.
        blur_mode (str, optional): Blur mode for preprocessing ('recursive2D' by default).
        deep_learning (str, optional): Model type for deep learning segmentation ('rapid' by default).
        threshold_spot_ch1 (float, optional): Threshold for first spot channel segmentation (default 0.5).
        threshold_nuclei (float, optional): Threshold for nuclei segmentation (default 0.5).
        threshold_spot_ch2 (float, optional): Threshold for second spot channel segmentation (default 0.5).
        RUN_SEG (bool, optional): If True, perform segmentation and spot detection (default True).
        RUN_REG (bool, optional): If True, perform registration (default False).
        RUN_DISTANCE_MEASUREMENTS (bool, optional): If True, compute distances and apply corrections (default True).
        RUN_CTRLS (bool, optional): If True, generate control images for visual inspection (default True).
        list_pairs_for_reg (list, optional): List of point pairs used for registration (default None).

    Returns:
        None: Saves results, plots, and control images to disk.

    Example:
        ```python
        paths = ["dataset1.tif", "dataset2.tif"]
        correction_matrix_save_path = "corrections/"
        run_analysis(
            paths,
            correction_matrix_save_path,
            PAIRING_THRESHOLD=50,
            ch_nuclei=0,
            first_spot_channel=1,
            second_spot_channel=2,
            RUN_SEG=True,
            RUN_REG=True,
            RUN_DISTANCE_MEASUREMENTS=True,
            RUN_CTRLS=True
        )

        ```
    """

    start = timer()
    print('Analysis started')

    if not isinstance(second_spot_channel, list):
        second_spot_channels = [second_spot_channel]
    else:
        second_spot_channels = second_spot_channel

    # --- Segmentation and spot detection ---
    if RUN_SEG or RUN_GAUSSIAN_FIT:
        if RUN_SEG:
            segment_spots_and_nuclei(
                paths,
                ch_nuclei=ch_nuclei,
                first_spot_channel=first_spot_channel,
                second_spot_channel=second_spot_channel,
                channels_to_blur=channels_to_blur,
                blur_mode=blur_mode,
                deep_learning=deep_learning,
                nuclear_model_to_use = nuclear_model_to_use,
                spot_model_to_use = spot_model_to_use
            )

        if RUN_GAUSSIAN_FIT:
            detect_spots_and_nuclei(
                paths,
                ch_nuclei=ch_nuclei,
                first_spot_channel=first_spot_channel,
                second_spot_channel=second_spot_channel,
                # spot_channels=,
                area_threshold=area_threshold,
                channels_to_blur=channels_to_blur,
                blur_mode=blur_mode
            )


        pair_spots(paths, PAIRING_THRESHOLD=PAIRING_THRESHOLD, first_spot_channel=first_spot_channel, second_spot_channels=second_spot_channel)

    # --- Correction and analysis of green/blue spots ---
    if True:  # TODO --> maybe put this as a parameter
        # try:
        #
        #     # compute registration center on 0
        #     MASK = True
        #     order = ['z', 'y', 'x']   # consistent coordinate order
        #     blue_spots, green_spots, tags = get_green_and_blue_dots(src=paths, order=order, TAG=MASK)
        #     # Call the correction function
        #     corrected_spots = perform_correction(blue_spots, green_spots)
        #     finalize_analysis_and_save_db(blue_spots, corrected_spots, green_spots, tags, order=order, MASK=MASK)
        # except:
        #     print('an error occurred with the gregor method, maybe empty files')
        #     traceback.print_exc()
        try:
            # Loop through each path
            for path in paths:
                try:
                    # Get the database path
                    db_path = smart_name_parser(path, 'TA') + '/FISH.db'

                    # Connect to the database to find tables
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    # Get all tables that match the pattern
                    cursor.execute("""
                                   SELECT name
                                   FROM sqlite_master
                                   WHERE type = 'table'
                                     AND name LIKE 'points_n_distances3D_only_in_nuclei%'
                                       AND name NOT LIKE '%acc%'
                                   """)
                    tables = [table[0] for table in cursor.fetchall()]
                    cursor.close()
                    conn.close()

                    # Process each table
                    for table in tables:
                        try:
                            # Set up parameters
                            MASK = True
                            order = ['z', 'y', 'x']  # consistent coordinate order

                            # Get spots for this specific table
                            blue_spots, green_spots, tags = get_green_and_blue_dots(
                                src=[path],
                                order=order,
                                TAG=MASK,
                                table_name=table
                            )

                            # Skip if no spots found
                            if len(blue_spots) == 0 or len(green_spots) == 0:
                                print(f"No spots found in table {table}, skipping")
                                continue

                            # Call the correction function
                            corrected_spots = perform_correction(blue_spots, green_spots)

                            # Generate output filename based on table name
                            # output_filename = smart_name_parser(
                            #     path,
                            #     f'{table}_null_vector_method'
                            # )


                            # Finalize analysis and save to DB with the output filename
                            finalize_analysis_and_save_db(
                                blue_spots,
                                corrected_spots,
                                green_spots,
                                tags,
                                order=order,
                                MASK=MASK,
                                output_filename=f'{table}_lcc'
                            )

                        except:
                            print('Error processing table')
                            traceback.print_exc()

                except:
                    print(f'Error accessing database for path {path}')
                    traceback.print_exc()

        except:
            print('An error occurred with the registration method')
            traceback.print_exc()

    # --- Registration ---
    if RUN_REG:
        USE_AFFINE_TRAFO_WITH_SHEAR = True
        # Convert second_spot_channel to a list if it's not already


        # Process each channel in the list
        for ch2 in second_spot_channels:
            # Convert negative channel index to positive

                db_to_read = 'points_n_distances3D_only_in_nuclei'
                # First try with the 'only_in_nuclei' table
                try:
                    compute_affine_transform_for_images(
                        list_pairs_for_reg,
                        correction_matrix_save_path,
                        ch1=first_spot_channel,
                        ch2=ch2,
                        db_to_read=db_to_read,  # This will be used as a prefix
                        USE_AFFINE_TRAFO_WITH_SHEAR=USE_AFFINE_TRAFO_WITH_SHEAR
                    )
                except:
                    try:
                        db_to_read = 'points_n_distances3D'
                        compute_affine_transform_for_images(
                            list_pairs_for_reg,
                            correction_matrix_save_path,
                            ch1=first_spot_channel,
                            ch2=ch2,
                            db_to_read=db_to_read,  # This will be used as a prefix
                            USE_AFFINE_TRAFO_WITH_SHEAR=USE_AFFINE_TRAFO_WITH_SHEAR,
                            # Add a parameter to indicate we don't want the 'only_in_nuclei' version
                        )
                    except Exception as e:
                        print(f"Failed to compute affine transform for channels {first_spot_channel} and {ch2}: {str(e)}")
                        traceback.print_exc()

    # --- Apply affine transform and measure distances ---
    if RUN_DISTANCE_MEASUREMENTS:
        table_prefixes = [
            'points_n_distances3D',
            'points_n_distances3D_only_in_nuclei'
        ]
        for ch2 in second_spot_channels:
            for table_prefix in table_prefixes:

                # Construct the actual table name using the channel numbers
                # table_name = f"{table_prefix}_ch{second_spot_channel}_ch{second_spot_channel}"
                try:
                    apply_affine_transform_to_all_images(paths, correction_matrix_save_path, table_prefix,  ch1=first_spot_channel,
                        ch2=ch2)
                except:
                    print(f'error applying reg for channels ch{first_spot_channel} and ch{ch2}')
                    traceback.print_exc()

            # # Optional: human curated distances
            # try:
            #     table = 'human_curated_distances_3D'
            #     apply_affine_transform_to_all_images(paths, correction_matrix_save_path, table)
            # except:
            #     traceback.print_exc()
            #     print('No human curated file found')

            # Optional: null vector method for chromatic aberration correction
            try:
                table = f'points_n_distances3D_only_in_nuclei_ch{first_spot_channel}_ch{ch2}_lcc'
                for path in paths:
                    try:
                        distances = get_column_from_sqlite_table(
                            smart_name_parser(path, 'TA') + '/FISH.db',
                            table,
                            'distance'
                        )
                        distances = np.array(distances, dtype=float)
                        plot_distance_distribution(
                            distances,
                            smart_name_parser(path, 'TA') + '/' + table + '.png'
                        )
                    except:
                        traceback.print_exc()
                        print('Null vector plotting failed for', path)
            except:
                traceback.print_exc()
                print('Error in null vector method analysis')

    # --- Plotting final results (violin plots) ---
    if True:
        try:
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d_%H-%M-%S_')
            parent = smart_name_parser(paths[0], 'parent')

            # Define base table names
            # base_table_names = [
            #     'points_n_distances3D_only_in_nuclei_acc',
            #     'points_n_distances3D_only_in_nuclei_lcc'
            # ]

            for ch2 in second_spot_channels:

                plot_analysis(paths,
                              group_files_by_name_similarity=True,
                              output_file_name=os.path.join(parent, date_str + f'violin_plot_ch{first_spot_channel}_ch{ch2}.pdf'),
                              table_names=[
                                  # 'human_curated_distances_3D_acc',
                                  f'points_n_distances3D_only_in_nuclei_ch{first_spot_channel}_ch{ch2}_acc',
                                  f'points_n_distances3D_only_in_nuclei_ch{first_spot_channel}_ch{ch2}_lcc'])

                print('saving violin plot to', os.path.join(parent, date_str + f'violin_plot_ch{first_spot_channel}_ch{ch2}.pdf'))
        except:
            traceback.print_exc()
            print('error in plot')

    # --- Controls ---
    if RUN_CTRLS:
        # Optional: plot pairs directly on images
        for ch2 in second_spot_channels:
            plot_all_paired_spots(
                paths, ch1=first_spot_channel, ch2=ch2
                # ['points_n_distances3D_only_in_nuclei', 'points_n_distances3D']
            )
        for ch2 in second_spot_channels:
            do_controls_for_easy_check(paths, ch_nuclei=ch_nuclei, ch1=first_spot_channel, ch2=ch2)

    print('total time', timer() - start)


# TODO --> try gaussian fit on LOG
# NB blurring seems to be magic --> I now can suddenly detect the spots of Sarah with high precision and I don't miss any too
# maybe start to do an interface with the stuff

# from skimage import filters
def blur_3D(single_channel_image, mode='real 3D', sigma=2.):
    # https://stackoverflow.com/questions/45723088/how-to-blur-3d-array-of-points-while-maintaining-their-original-values-python

    # TODO check that gaussian_laplace --> is that the laplacian of gaussian ????
    if '3D' in mode:
        from scipy.ndimage import gaussian_filter
        # NB maybe I blur too much in the Z dimension comapred to x-y --> maybe I should take this into account by caring about spot size or pixel size

        # do a real 3D blur (maybe ask for blur as a parameter)
        # single_channel_image = filters.gaussian(single_channel_image, gaussian_blur, preserve_range=True, mode='wrap') # deprecated ???
        # single_channel_image = gaussian_filter(single_channel_image, sigma,  mode='wrap')
        single_channel_image = gaussian_filter(single_channel_image, sigma, mode='reflect')
        # single_channel_image = gaussian_filter(single_channel_image, (0.75,1.6,1.6),  mode='reflect') # test -->
        print('3D blur')
    elif 'dog' in mode.lower():
        # try log
        if '3D' in mode:
            # https://stackoverflow.com/questions/22050199/python-implementation-of-the-laplacian-of-gaussian-edge-detection
            from skimage.filters import difference_of_gaussians
            single_channel_image = difference_of_gaussians(single_channel_image, sigma / 1.6,
                                                           sigma)  # approx of laplacian of gaussian
        else:
            raise Exception('not implemented yet')
    elif 'log' in mode.lower():
        from scipy.ndimage import gaussian_laplace
        if '3D' in mode:
            single_channel_image = gaussian_laplace(single_channel_image, sigma)
        else:
            raise Exception('not implemented yet')
    else:
        from scipy.ndimage import gaussian_filter
        # blur every 2D image recursively
        for iii, img in enumerate(single_channel_image):
            # single_channel_image[iii]=filters.gaussian(img, gaussian_blur, preserve_range=True, mode='wrap')
            # single_channel_image[iii]=gaussian_filter(img, sigma, mode='wrap')
            single_channel_image[iii] = gaussian_filter(img, sigma, mode='reflect')
        print('2D recursive blur')

    return single_channel_image


def plot_violin_with_median(data, **kwargs):
    """
    Create a violin plot for the 'distance' column of a DataFrame and annotate it with the median.

    Args:
        data (pandas.DataFrame): DataFrame containing a 'distance' column to visualize.
        **kwargs: Additional keyword arguments passed to seaborn.violinplot (e.g., palette, color, scale).

    Returns:
        None: The plot is drawn on the current Matplotlib axis.

    Example:
        ```python
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame({'distance': [1.2, 2.5, 1.8, 3.0, 2.2]})
        plot_violin_with_median(df, palette='muted')
        plt.show()

        ```
    """

    # Get the current axis (allows function to be embedded in subplots)
    ax = plt.gca()

    # Create the violin plot on the 'distance' column
    sns.violinplot(
        data=data,
        y="distance",
        bw=0.08,  # KDE bandwidth; tweakable depending on noise
        ax=ax,
        **kwargs
    )

    # Compute median and sample size
    median = data["distance"].median()
    count = len(data)

    # Overlay horizontal line showing the median
    ax.axhline(
        median,
        color="red",
        linestyle="--",
        label=f"Median: {median:.3f} (n={count})"
    )

    # Display legend for clarity
    ax.legend()


def plot_analysis(
    paths,
    distance_cut_off=2.5,
    group_files_by_name_similarity=False,
    output_file_name=None,
    table_names=[
        # 'human_curated_distances_3D_acc',
        'points_n_distances3D_only_in_nuclei_acc',
        'points_n_distances3D_only_in_nuclei_lcc'
    ]
):
    """
    Generate violin plots of distances from FISH analysis datasets with optional grouping.

    This function loads distance data from multiple FISH.db files, filters by a
    cutoff distance, optionally groups datasets by filename similarity, and
    generates violin plots with median annotations.

    Args:
        paths (list of str): List of paths to FISH datasets or directories containing 'FISH.db'.
        distance_cut_off (float, optional): Maximum distance (µm) to include in the plot. Default is 2.5.
        group_files_by_name_similarity (bool, optional): If True, group datasets based on filename similarity. Default is False.
        output_file_name (str or None, optional): Path to save the PDF of the plot. If None, displays the plot. Default is None.
        table_names (list of str, optional): List of database table names to extract distances from. Default is the two CHAB-corrected tables.

    Returns:
        None: The plot is either displayed or saved to a file.

    Example:
        ```python
        paths = ['dataset1', 'dataset2']
        plot_analysis(paths, distance_cut_off=3.0, group_files_by_name_similarity=True,
                      output_file_name='violin_plot.pdf')

        ```
    """

    # Ensure paths point to FISH.db
    paths = [
        path if path.endswith('FISH.db') else os.path.join(smart_name_parser(path, 'TA'), 'FISH.db')
        for path in paths
    ]
    print('Resolved FISH.db paths:', paths)
    print('table_names', table_names)

    # Combine distance data from all databases and tables
    header, data = combine_single_file_queries(
        paths,
        'SELECT distance FROM ',
        table_names=table_names,
        db_name=None,
        output_filename=None,
        return_header=True
    )

    # Create a pandas DataFrame
    df = pd.DataFrame(data=data if data else None, columns=header)

    # --- Optional grouping by filename similarity ---
    if group_files_by_name_similarity:
        from batoolset.utils.loadlist import loadlist
        from sklearn.cluster import AgglomerativeClustering

        filenames = list(set(df['filename'].tolist()))
        if filenames[0].endswith('FISH.db'):
            filenames = [smart_name_parser(file, 'parent') for file in filenames]

        # Extract base names
        base_names = [_extract_base_name(f) for f in filenames]

        # Compute Levenshtein distance matrix
        n = len(base_names)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = levenshtein_distance(base_names[i], base_names[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # symmetric

        if n >= 2:
            clustering = AgglomerativeClustering(
                metric='precomputed',
                linkage='complete',
                distance_threshold=10,
                n_clusters=None
            )
            clustering.fit(dist_matrix)

            grouped_files = {}
            for i, label in enumerate(clustering.labels_):
                grouped_files.setdefault(label, []).append(filenames[i])

            # Assign group names based on longest common substring
            for label, files in grouped_files.items():
                group = longest_common_substring(files, use_base_name=True)
                for f in files:
                    df.loc[df['filename'].str.contains(f, regex=False), 'group'] = group

    # Determine if grouping is active
    GROUP_MODE = 'group' in df.columns and df['group'].nunique() > 1

    # --- Filter by distance cutoff ---
    if distance_cut_off and distance_cut_off > 0:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        print('Total entries before cutoff:', len(df))
        df = df[df['distance'] < distance_cut_off]
        print('Total entries after cutoff:', len(df))

    # --- Create violin plot ---
    fig, ax = plt.subplots(figsize=(16, 16))

    if GROUP_MODE:
        sns.violinplot(
            data=df,
            y='distance',
            x='group',
            hue='group',
            bw_method=0.08,
            split=False,
            ax=ax
        )
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Overlay median line for each group
        positions = ax.get_xticks()
        for i, group in enumerate(df['group'].unique()):
            group_data = df[df['group'] == group]
            median = group_data['distance'].median()
            count = len(group_data)
            violin_width = ax.collections[i].get_paths()[0].get_extents().width
            random_color = np.random.rand(3,)
            ax.plot(
                [positions[i] - violin_width / 2, positions[i] + violin_width / 2],
                [median, median],
                color=random_color,
                linestyle='--',
                label=f'{group} Median: {median:.3f} (n={count})'
            )
    else:
        sns.violinplot(data=df, y='distance', bw_method=0.08, split=False, ax=ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        median = df['distance'].median()
        count = len(df)
        ax.axhline(median, color='red', linestyle='--', label=f'Median: {median:.3f} (n={count})')

    # Add legend above plot
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1))

    # Show or save plot
    if output_file_name is None:
        plt.show()
    else:
        plt.savefig(output_file_name, format='pdf', bbox_inches='tight')

    plt.close('all')


if __name__ == '__main__':
    import sys

    if True:
        paths = loadlist('/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048/*.czi')
        if True:  # TODO --> maybe put this as a parameter
            try:
                # compute registration center on 0
                MASK = True
                order = ['z', 'y', 'x']  # consistent coordinate order
                blue_spots, green_spots, tags = get_green_and_blue_dots(src=paths, order=order, TAG=MASK)
                # Call the correction function
                corrected_spots = perform_correction(blue_spots, green_spots)
                finalize_analysis_and_save_db(blue_spots, corrected_spots, green_spots, tags, order=order, MASK=MASK)
            except:
                print('an error occurred with the gregor method, maybe empty files')
                traceback.print_exc()

        sys.exit(0)

    # if True:
    #     paths = loadlist('/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048/*.czi')
    #
    #
    #
    #     sys.exit(0)
    #
    #
    # if True:
    #     paths = loadlist('/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048/*.czi')
    #     if True:
    #         try:
    #             from datetime import datetime
    #
    #             now = datetime.now()
    #             date_str = now.strftime('%Y-%m-%d_%H-%M-%S_')
    #             parent = smart_name_parser(paths[0], 'parent')
    #
    #             plot_analysis(paths,
    #                           group_files_by_name_similarity=True,
    #                           output_file_name=os.path.join(parent, date_str + 'violin_plot.pdf'),
    #                           table_names=[
    #                               'human_curated_distances_3D_acc',
    #                               'points_n_distances3D_only_in_nuclei_acc',
    #                               'points_n_distances3D_only_in_nuclei_lcc'])
    #
    #             print('saving violin plot to', os.path.join(parent, date_str + 'violin_plot.pdf'))
    #         except:
    #             traceback.print_exc()
    #             print('error in plot')
    #     sys.exit(0)

    if True:
        raise Exception('from now on always run your code through the tst_contab.py file especially because of channel change of Manue')
        sys.exit(0)