import numpy as np
from sklearn.linear_model import LinearRegression
from batoolset.img import get_voxel_conversion_factor
from batoolset.nps.tools import find_factors
from batoolset.ta.database.sql import create_table_and_append_data
from batoolset.ta.measurements.TAmeasures import distance_between_points
from batoolset.files.tools import smart_name_parser
from fishdist.spot_data_utils import compute_and_plot_points_for_all_dims


# from gergor_tools_bioengeneering import compute_and_plot_points_for_all_dims

def perform_correction(source_points, target_points, recenter=True):
    """
    Apply linear bias correction to 3D points by aligning one set of points to another.

    This function performs a per-axis linear correction to reduce systematic
    biases between two sets of corresponding 3D points obtained from the same image.
    One set (`source_points`) is aligned to the other (`target_points`) using slope
    correction and optional recentering.

    The procedure is as follows:
    1. Compute the difference between each point in `source_points` and the corresponding
       point in `target_points`.
    2. Subtract the mean difference to center the distribution at zero.
    3. Fit a linear regression per axis to estimate scaling (slope) biases.
    4. Adjust `source_points` using the per-axis slopes.
    5. Optionally remove any residual global translation to recenter the corrected points.

    Args:
        source_points (np.ndarray of shape (N, 3)): 3D coordinates of the points to be corrected.
        target_points (np.ndarray of shape (N, 3)): 3D coordinates of the points used for alignment.
        recenter (bool, optional): If True, subtracts residual translation to recenter
            the corrected points. Default is True.

    Returns:
        np.ndarray of shape (N, 3): Corrected 3D points after linear slope correction
        and optional recentering.

    Example:
        ```python
        import numpy as np

        source = np.array([[1.0, 2.0, 3.0],
                           [2.0, 3.0, 4.0]])
        target = np.array([[0.9, 2.1, 2.9],
                           [2.1, 2.9, 4.1]])

        corrected = perform_correction(source, target)
        print(corrected)
        ```
    """

    # We assume 3D inputs (x, y, z)
    num_dimensions = 3

    # Prepare a diagonal correction matrix (slopes for each axis)
    correction_matrix = np.zeros((num_dimensions, num_dimensions))


    # Difference between measured and reference points
    diff = source_points - target_points

    # Remove mean displacement before slope estimation
    mean_diff = np.mean(diff, axis=0)
    diff = diff - mean_diff

    # Fit per-axis linear regression to estimate scaling correction
    for dim in range(num_dimensions):
        model = LinearRegression()
        model.fit(
            source_points[:, dim].reshape(-1, 1),
            diff[:, dim].reshape(-1, 1)
        )
        slope = model.coef_[0][0]
        correction_matrix[dim, dim] = slope

    # Compute correction vector
    correction_vector = np.dot(source_points, correction_matrix)

    # Apply the correction
    corrected_points = source_points - correction_vector

    # Optionally recenter
    if recenter:
        residual_shift = np.mean(corrected_points - target_points, axis=0)
        corrected_points -= residual_shift

    return corrected_points


def finalize_analysis_and_save_db(source_points, corrected_spots, target_points, tags, order=['x','y','z'], MASK=True, output_filename=None):
    """
      Compute distances between corrected points and a target set, save results, and generate plots.

      This function performs a per-tag (or global) analysis of 3D points:
      - Groups points by `tags` (if MASK=True).
      - Computes 3D distances between `corrected_spots` and `target_points`.
      - Saves per-tag results in a database with one row per point pair.
      - Generates dimension-wise plots showing alignment before and after correction.

      Args:
          source_points (np.ndarray of shape (N, 3)): Original points before correction.
          corrected_spots (np.ndarray of shape (N, 3)): Points after linear correction.
          target_points (np.ndarray of shape (N, 3)): Target points used for alignment.
          tags (np.ndarray of shape (N,)): Labels used to group points for per-tag analysis.
          order (list of str, optional): Dimension names for database columns. Default is ['x','y','z'].
          MASK (bool, optional): If True, perform per-tag analysis; otherwise, process all points together. Default is True.
          output_filename (str, optional): Optional output filename for the database.

      Returns:
          None: Results are saved to per-tag databases, and plots are generated; no values are returned.

      Example:
          ```python
          import numpy as np

          source = np.array([[1,2,3],[4,5,6]])
          corrected = np.array([[1.1,2.0,2.9],[3.9,5.1,6.0]])
          target = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
          tags = np.array(['A','B'])

          finalize_analysis_and_save_db(source, corrected, target, tags)
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
            distances = distance_between_points(corrected_spots[tag_mask], target_points[tag_mask],
                                               rescaling_factor=voxel_scale)
            distances = distances[..., np.newaxis]  # convert to column vector

            # Concatenate corrected points, reference points, and distances
            data = np.concatenate((corrected_spots[tag_mask], target_points[tag_mask], distances), axis=1)

            # Save data to per-tag database
            db_name = smart_name_parser(factor, 'TA') + '/FISH.db'
            # table_name = "points_n_distances3D_only_in_nuclei_lcc"
            create_table_and_append_data(db_name, output_filename, columns, data, temporary=False)

            # Generate plots for each dimension
            compute_and_plot_points_for_all_dims(source_points[tag_mask], corrected_spots[tag_mask],
                                                 target_points[tag_mask], rescaling_factor=voxel_scale)

