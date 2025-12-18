import numpy as np
from sklearn.linear_model import LinearRegression
from batoolset.img import get_voxel_conversion_factor
from batoolset.nps.tools import find_factors
from batoolset.ta.database.sql import create_table_and_append_data
from batoolset.ta.measurements.TAmeasures import distance_between_points
from batoolset.files.tools import smart_name_parser
from fishdist.spot_data_utils import compute_and_plot_points_for_all_dims


# from gergor_tools_bioengeneering import compute_and_plot_points_for_all_dims

def perform_correction(measured_points, reference_points, recenter=True):
    """
    Apply linear bias correction to 3D points based on a reference set.

    This function computes per-axis slope deviations between `measured_points` and
    `reference_points`, corrects the measured points, and optionally recenters them
    to match the reference distribution.

    Args:
        measured_points (np.ndarray of shape (N, 3)): The measured or distorted 3D centroid positions.
        reference_points (np.ndarray of shape (N, 3)): The reference 3D centroid positions.
        recenter (bool, optional): If True, subtracts residual translation to recenter corrected points. Default is True.

    Returns:
        np.ndarray of shape (N, 3): Corrected 3D points after slope correction and optional recentering.

    Example:
        ```python
        import numpy as np

        measured = np.array([[1.0, 2.0, 3.0],
                             [2.0, 3.0, 4.0]])
        reference = np.array([[0.9, 2.1, 2.9],
                              [2.1, 2.9, 4.1]])

        corrected = perform_correction(measured, reference)
        print(corrected)
        ```
    """

    # We assume 3D inputs (x, y, z)
    num_dimensions = 3

    # Prepare a diagonal correction matrix (slopes for each axis)
    correction_matrix = np.zeros((num_dimensions, num_dimensions))


    # Difference between measured and reference points
    diff = measured_points - reference_points

    # Remove mean displacement before slope estimation
    mean_diff = np.mean(diff, axis=0)
    diff = diff - mean_diff

    # Fit per-axis linear regression to estimate scaling correction
    for dim in range(num_dimensions):
        model = LinearRegression()
        model.fit(
            measured_points[:, dim].reshape(-1, 1),
            diff[:, dim].reshape(-1, 1)
        )
        slope = model.coef_[0][0]
        correction_matrix[dim, dim] = slope

    # Compute correction vector
    correction_vector = np.dot(measured_points, correction_matrix)

    # Apply the correction
    corrected_points = measured_points - correction_vector

    # Optionally recenter
    if recenter:
        residual_shift = np.mean(corrected_points - reference_points, axis=0)
        corrected_points -= residual_shift

    return corrected_points


def finalize_analysis_and_save_db(measured_points, corrected_spots, reference_points, tags, order=['x','y','z'], MASK=True):
    """
    Compute distances between corrected and reference 3D points, save results per tag, and generate plots.

    This function groups points by `tags` (if MASK is True), calculates 3D distances between
    corrected and reference points, stores the results in per-tag databases, and produces
    dimension-wise plots.

    Args:
        measured_points (np.ndarray of shape (N, 3)): Original measured spot coordinates.
        corrected_spots (np.ndarray of shape (N, 3)): Corrected spot coordinates.
        reference_points (np.ndarray of shape (N, 3)): Reference spot coordinates.
        tags (np.ndarray of shape (N,)): Categorical labels used to group points.
        order (list of str, optional): Dimension names for database columns. Default is ['x','y','z'].
        MASK (bool, optional): If True, perform per-tag analysis; otherwise, process all points together. Default is True.

    Returns:
        None: Results are saved to databases and plots are generated; no value is returned.

    Example:
        ```python
        import numpy as np

        measured = np.array([[1,2,3],[4,5,6]])
        corrected = np.array([[1.1,2.0,2.9],[3.9,5.1,6.0]])
        reference = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
        tags = np.array(['A','B'])

        finalize_analysis_and_save_db(measured, corrected, reference, tags)
        ```
    """

    if MASK:
        # Gather unique tags for analysis
        unique_tags = find_factors(tags)

        # Prepare database column names: x1,y1,z1,x2,y2,z2,distance
        columns = [dim+"1" for dim in order] + [dim+"2" for dim in order] + ['distance']

        for factor in unique_tags:
            # Create boolean mask for current tag
            tag_mask = np.squeeze(tags == factor)

            # Compute voxel rescaling factor for this tag
            voxel_scale = get_voxel_conversion_factor(factor)

            # Compute distances between corrected and reference points
            distances = distance_between_points(corrected_spots[tag_mask], reference_points[tag_mask],
                                               rescaling_factor=voxel_scale)
            distances = distances[..., np.newaxis]  # convert to column vector

            # Concatenate corrected points, reference points, and distances
            data = np.concatenate((corrected_spots[tag_mask], reference_points[tag_mask], distances), axis=1)

            # Save data to per-tag database
            db_name = smart_name_parser(factor, 'TA') + '/FISH.db'
            table_name = "points_n_distances3D_only_in_nuclei_chromatic_aberrations_corrected_null_vector_method"
            create_table_and_append_data(db_name, table_name, columns, data, temporary=False)

            # Generate plots for each dimension
            compute_and_plot_points_for_all_dims(measured_points[tag_mask], corrected_spots[tag_mask],
                                                 reference_points[tag_mask], rescaling_factor=voxel_scale)

