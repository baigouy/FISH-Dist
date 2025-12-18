# basic tools for testing the chromatic aberration stuff

import sqlite3
import traceback

import numpy as np
import matplotlib.pyplot as plt

from batoolset.img import get_voxel_conversion_factor
from batoolset.files.tools import smart_name_parser
from batoolset.ta.measurements.TAmeasures import distance_between_points
from batoolset.utils.loadlist import loadlist

from fishdist.wing_analysis_utils import get_q1_q3

from batoolset.tools.logger import TA_logger  # logging

logger = TA_logger()  # logging_level=TA_logger.DEBUG

# Set the number of decimal places to check after the decimal point

# Calculate the corresponding atol value



# TODO --> add a tagging so that they can be grouped
def get_green_and_blue_dots(src='coloc', order=['x','y','z'], TAG=False,
                            check_voxel_size=True, atol=1e-6):
    """
    Load measured (blue) and reference (green) 3D spot coordinates from FISH datasets, optionally returning tags.

    This function extracts 3D spot coordinates from database files or a list of file paths, checks for
    consistent voxel sizes, and optionally returns the source tag for each point.

    Args:
        src (str or list of str, optional): Source of the data. Can be a string key ('coloc', 'paired', 'all')
            or a list of file paths. Default is 'coloc'.
        order (list of str, optional): Order of axes for database queries. Default is ['x','y','z'].
        TAG (bool, optional): If True, returns an additional array of tags corresponding to the source of each point. Default is False.
        check_voxel_size (bool, optional): If True, ensures all files have the same voxel size. Default is True.
        atol (float, optional): Tolerance for voxel size comparison. Default is 1e-6.

    Returns:
        blue_spots (np.ndarray of shape (N, 3)): Measured (blue) 3D coordinates.
        green_spots (np.ndarray of shape (N, 3)): Reference (green) 3D coordinates.
        tags (np.ndarray of shape (N, 1), optional): Returned only if TAG=True; contains file/source tags for each point.

    Example:
        ```python
        blue, green, tags = get_green_and_blue_dots(src='coloc', TAG=True)
        blue, green = get_green_and_blue_dots(src=['file1.czi', 'file2.czi'])
        ```
    """

    if isinstance(src, str):


        if False:
            # Connect to the SQLite database
            if src == 'coloc':
                db_path = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/230314 Coloc X2 ZH-2A m 2048_2023_03_14__17_43_51(1)/FISH.db'
            elif src == 'paired':
                db_path = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 25kby X2/X1 633 X2 565 yx1 yx2 m 12ng ul probes normal fix Hoe 1 4000 2/FISH.db'
            else:
                db_path = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 or CS/X1 633 X2 565 CantonS m S1/FISH.db'
                db_path = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 or CS/X1 633 X2 565 CantonS m NS2/FISH.db'
            all_colocs_dbs = [db_path]
        else:
            # Connect to the SQLite database
            if src == 'coloc':
                all_colocs = loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/*.czi')
            elif src == 'paired':
                all_colocs = loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 25kby X2/*.czi')
            elif src == 'all':
                # most likely tag should be set to True
                all_colocs=loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/*.czi')+loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 25kby X2/*.czi')+loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 or CS/*.czi')
                if not TAG:
                    print('TAG is set to False, so all images will be treated as one image, please make sure it makes sense')
            else:
                all_colocs = loadlist('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit X1 X2 late pupal wings/25kby X1 or CS/*.czi')
    else:
        # User passed a list of files directly
        all_colocs = src


    # Check voxel sizes for consistency
    if check_voxel_size:
        voxel_size = None
        for idx, file in enumerate(all_colocs):
            current_voxel = get_voxel_conversion_factor(file)
            if idx == 0:
                voxel_size = current_voxel
            else:
                if not np.allclose(list(current_voxel), list(voxel_size), atol=atol):
                    print(voxel_size, 'vs', current_voxel)
                    logger.error('Images do not have the same voxel size; skipping registration.')
                    return

    # Convert file names to database paths
    all_colocs_dbs = [smart_name_parser(name, 'TA') + '/FISH.db' for name in all_colocs]

    # Containers for all coordinates
    blue_spots_all = []
    green_spots_all = []
    tags_all = []

    # Load points from each database
    for file_idx, db_file in enumerate(all_colocs_dbs):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Build query for points in nuclei
            if order is not None:
                query = ('SELECT pt_1'+order[0]+'_px, pt_1'+order[1]+'_px, pt_1'+order[2]+'_px, '
                         'pt_2'+order[0]+'_px, pt_2'+order[1]+'_px, pt_2'+order[2]+'_px ' +
                         (', "'+all_colocs[file_idx]+'" AS filename' if TAG else '') +
                         ' FROM points_n_distances3D_only_in_nuclei')
            else:
                query = ('SELECT pt_1x_px, pt_1y_px, pt_1z_px, pt_2x_px, pt_2y_px, pt_2z_px ' +
                         (', "'+all_colocs[file_idx]+'" AS filename' if TAG else '') +
                         ' FROM points_n_distances3D_only_in_nuclei')

            cursor.execute(query)
            results = cursor.fetchall()

            # Handle tags
            if TAG:
                file_tags = np.array([[tag] for *_, tag in results])
                results = [res[:-1] for res in results]
                tags_all.append(file_tags)

            # Extract blue and green coordinates
            blue_spots = np.array([[float(z), float(y), float(x)] for z, y, x, _, _, _ in results])
            green_spots = np.array([[float(z), float(y), float(x)] for _, _, _, z, y, x in results])

            blue_spots_all.append(blue_spots)
            green_spots_all.append(green_spots)

            cursor.close()
            conn.close()
        except:
            logger.error('Error occurred loading database, likely empty file.')
            traceback.print_exc()

    # Concatenate all coordinates
    blue_spots = np.concatenate(blue_spots_all, axis=0)
    green_spots = np.concatenate(green_spots_all, axis=0)

    if TAG:
        tags = np.concatenate(tags_all, axis=0)
        return blue_spots, green_spots, tags
    else:
        return blue_spots, green_spots


def compute_pairwise_distance(coord_set1, coord_set2, rescaling_factor=None):
    """
    Compute pairwise distances between two sets of coordinates.

    For each index `i`, calculates the distance between `coord_set1[i]` and `coord_set2[i]`.
    If the sets have different lengths, computes distances up to the length of the shorter set
    and prints a warning.

    Args:
        coord_set1 (list or np.ndarray): First set of coordinates. Each element should be a point (array-like).
        coord_set2 (list or np.ndarray): Second set of coordinates. Each element should be a point (array-like).
        rescaling_factor (float or sequence, optional): Factor to scale coordinates before computing distances.
            Can be used to convert voxel coordinates to physical units. Default is None.

    Returns:
        list of float: Pairwise distances between corresponding points.

    Example:
        ```python
        >>> import numpy as np
        >>> set1 = np.array([[0,0],[1,1],[2,2]])
        >>> set2 = np.array([[1,0],[1,2],[2,3]])
        >>> compute_pairwise_distance(set1, set2)
        [1.0, 1.0, 1.0]

        >>> compute_pairwise_distance(set1, set2, rescaling_factor=2)
        [2.0, 2.0, 2.0]

        ```
    """


    distances = []

    # Determine number of pairs (use length of shorter set)
    n_pairs = len(coord_set1)
    if n_pairs != len(coord_set2):
        print(
            "Warning: the two sets have different sizes. "
            "You likely have unpaired sets; distances may be meaningless."
        )
        n_pairs = min(n_pairs, len(coord_set2))

    # Compute distance for each pair
    for i in range(n_pairs):
        distances.append(
            distance_between_points(
                coord_set1[i],
                coord_set2[i],
                rescaling_factor=rescaling_factor
            )
        )

    return distances

def compute_and_plot_points_for_all_dims(measured_points, corrected_spots, reference_points,
                                         rescaling_factor=[0.1037832, 0.1037832, 0.2500000]):
    """
    Compute and display statistics of 3D spot positions before and after correction.

    This function calculates the vectors from measured to reference points, computes
    pairwise distances, medians, and interquartile ranges (IQRs), and prints mean vectors
    before and after correction. Intended for quality assessment of chromatic aberration
    correction.

    Args:
        measured_points (ndarray of shape (N,3)): Original measured spot coordinates.
        corrected_spots (ndarray of shape (N,3)): Corrected spot coordinates after chromatic aberration correction.
        reference_points (ndarray of shape (N,3)): Reference spot coordinates.
        rescaling_factor (list or array-like of length 3, optional): Scaling factors for x, y, z dimensions
            to convert distances to physical units. Default is [0.1037832, 0.1037832, 0.25].

    Returns:
        None: Prints summary statistics (mean vectors, medians, IQRs) for before/after correction.

    Example:
        ```python
        >>> import numpy as np
        >>> measured = np.array([[0,0,0],[1,1,1],[2,2,2]])
        >>> corrected = np.array([[0,0,0],[1,1,0.9],[2,2,1.9]])
        >>> reference = np.array([[0,0,0],[1,1,1],[2,2,2]])
        >>> compute_and_plot_points_for_all_dims(measured, corrected, reference)
        Mean vector before correction: [0. 0. 0.]
        Mean vector after correction: [ 0.          0.         -0.06666667]
        Original distances (median * 1000 nm): 0.0 IQR: [0. 0.]
        Corrected distances (median * 1000 nm): 24.999999999999993 IQR: [0.0125 0.025 ]

        ```
    """

    # Vectors from measured to reference points before and after correction
    vector_before = measured_points - reference_points
    vector_after = corrected_spots - reference_points

    # Compute pairwise distances between corrected and reference points
    pairwise_distances_corrected = compute_pairwise_distance(corrected_spots, reference_points,
                                                             rescaling_factor=rescaling_factor)
    corrected_median = np.median(pairwise_distances_corrected)
    iqr_corrected = get_q1_q3(pairwise_distances_corrected)

    # Compute pairwise distances between original (measured) and reference points
    pairwise_distances_original = compute_pairwise_distance(measured_points, reference_points,
                                                            rescaling_factor=rescaling_factor)
    original_median = np.median(pairwise_distances_original)
    iqr_original = get_q1_q3(pairwise_distances_original)

    # Mean vectors before and after correction
    mean_vector_before = np.mean(vector_before, axis=0)
    mean_vector_after = np.mean(vector_after, axis=0)

    # Print results
    print("Mean vector before correction:", mean_vector_before)
    print("Mean vector after correction:", mean_vector_after)

    print("Original distances (median * 1000 nm):", original_median * 1000, "IQR:", iqr_original)
    print("Corrected distances (median * 1000 nm):", corrected_median * 1000, "IQR:", iqr_corrected)


def plot_points_3D(blue_spots, corrected_spots, green_spots):
    """
    Create a 3D scatter plot of blue, green, and corrected spot coordinates.

    This function visualizes three sets of 3D points: the original measured (blue),
    reference (green), and corrected (red) spots, allowing visual assessment of
    chromatic aberration correction.

    Args:
        blue_spots (ndarray of shape (N,3)): Original measured spot coordinates.
        corrected_spots (ndarray of shape (N,3)): Corrected spot coordinates after correction.
        green_spots (ndarray of shape (N,3)): Reference spot coordinates.

    Returns:
        None: Displays a 3D scatter plot of the points.

    Example:
        ```python
        >>> import numpy as np
        >>> measured = np.array([[0,0,0],[1,1,1],[2,2,2]])
        >>> corrected = np.array([[0,0,0],[1,1,0.9],[2,2,1.9]])
        >>> reference = np.array([[0,0,0],[1,1,1],[2,2,2]])
        >>> plot_points_3D(measured, corrected, reference)
        ```
    """

    # PLOTTING PART

    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # assume blue_spots and green_spots are the arrays of spot coordinates generated earlier
    # also assume corrected_spots is the array of corrected spot coordinates obtained from the code I provided earlier

    # create a 3D plot of the blue spots, green spots, and corrected spots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the blue spots
    ax.scatter(blue_spots[:, 0], blue_spots[:, 1], blue_spots[:, 2], c='blue', label='Blue spots')

    # plot the green spots
    ax.scatter(green_spots[:, 0], green_spots[:, 1], green_spots[:, 2], c='green', label='Green spots')

    # plot the corrected spots
    ax.scatter(corrected_spots[:, 0], corrected_spots[:, 1], corrected_spots[:, 2], c='red', label='Corrected spots')

    # set the axis labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # show the plot
    plt.show()



def plot_each_dim(blue_spots, corrected_spots, green_spots, axes_names=['X','Y','Z'], center_on_zero=True):
    """
    Plot displacement along each dimension before and after correction, with linear regression slopes.

    This function visualizes the per-dimension displacements of 3D spots relative to
    reference positions, comparing measured (blue) and corrected (red) spots. It optionally
    centers displacements on zero, fits linear regressions, and prints slope values.

    Args:
        blue_spots (ndarray of shape (N,3)): Original measured spot coordinates.
        corrected_spots (ndarray of shape (N,3)): Corrected spot coordinates after chromatic aberration correction.
        green_spots (ndarray of shape (N,3)): Reference spot coordinates.
        axes_names (list of str, optional): Names of axes to label plots. Default is ['X','Y','Z'].
        center_on_zero (bool, optional): If True, centers displacements by subtracting the mean. Default is True.

    Returns:
        None: Displays 2D scatter plots with regression lines and prints slope statistics.

    Example:
        ```python
        import numpy as np
        measured = np.array([[0,0,0],[1,1,1],[2,2,2]])
        corrected = np.array([[0,0,0],[1,0.9,1],[2,1.9,2]])
        reference = np.array([[0,0,0],[1,1,1],[2,2,2]])
        plot_each_dim(measured, corrected, reference)

        ```
    """

    # assume blue_spots, green_spots, and corrected_spots are the arrays of spot coordinates generated earlier

    # calculate the displacement before and after correction in each dimension
    displacement_before = blue_spots - green_spots
    displacement_after = corrected_spots - green_spots

    if center_on_zero:
        displacement_before = displacement_before - np.mean(displacement_before, axis=0)
        displacement_after = displacement_after - np.mean(displacement_after, axis=0)

    # create three 2D plots, one for each dimension (x, y, and z)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))



    for dim in range(displacement_before.shape[-1]):
        # plot the displacement in the x dimension as a function of spot position
        axs[dim].scatter(blue_spots[:, dim], displacement_before[:, dim], c='blue', label='Before correction', alpha=0.1)
        axs[dim].scatter(blue_spots[:, dim], displacement_after[:, dim], c='red', label='After correction', alpha=0.1)
        axs[dim].set_xlabel(axes_names[dim]+' position (pixels)')
        axs[dim].set_ylabel(axes_names[dim]+' displacement (pixels)')
        axs[dim].legend()

        from sklearn.linear_model import LinearRegression



        # Calculate the slope for the points before correction
        x_before = blue_spots[:, dim].reshape(-1, 1)
        y_before = displacement_before[:, dim].reshape(-1, 1)
        regression_before = LinearRegression()
        regression_before.fit(x_before, y_before)
        slope_before = regression_before.coef_[0][0]

        # Print the readjusted slopes
        print("Slope before:")
        print(axes_names[dim]+" Dimension: {:.2f}".format(slope_before))

        # Calculate the slope for the points after correction
        x_after = blue_spots[:, dim].reshape(-1, 1)
        y_after = displacement_after[:, dim].reshape(-1, 1)
        regression_after = LinearRegression()
        regression_after.fit(x_after, y_after)
        slope_after = regression_after.coef_[0][0]

        # Print the readjusted slopes
        print("Readjusted Slopes:")
        print(axes_names[dim]+" Dimension: {:.2f}".format(slope_after))

        # Plot the lines corresponding to the slopes
        axs[dim].plot(x_before, regression_before.predict(x_before), 'lightblue', label='Slope Before: {:.2f}'.format(slope_before))
        # axs[dim].plot(x_before, regression_before.predict(x_before), 'b-', label='Slope Before: {:.2f}'.format(slope_before))
        axs[dim].plot(x_after, regression_after.predict(x_after), 'salmon', label='Slope After: {:.2f}'.format(slope_after))
        # axs[dim].plot(x_after, regression_after.predict(x_after), (1.0, 0.75, 0.8), label='Slope After: {:.2f}'.format(slope_after))

        # that does seem to work in a way

        axs[dim].set_xlabel(axes_names[dim] + ' Position')
        axs[dim].set_ylabel('Displacement')
        axs[dim].legend()
    #
    # # plot the displacement in the y dimension as a function of spot position
    # axs[1].scatter(blue_spots[:, 1], displacement_before[:, 1], c='blue', label='Before correction')
    # axs[1].scatter(blue_spots[:, 1], displacement_after[:, 1], c='red', label='After correction')
    # axs[1].set_xlabel('Y position (pixels)')
    # axs[1].set_ylabel('Y displacement (pixels)')
    # axs[1].legend()
    #
    # # plot the displacement in the z dimension as a function of spot position
    # axs[2].scatter(blue_spots[:, 0], displacement_before[:, 0], c='blue', label='Before correction')
    # axs[2].scatter(blue_spots[:, 0], displacement_after[:, 0], c='red', label='After correction')
    # axs[2].set_xlabel('Z position (pixels)')
    # axs[2].set_ylabel('Z displacement (pixels)')
    # axs[2].legend()



    # axs[0].set_xlabel(positions[i] + ' Position')
    # axs[0].set_ylabel('Displacement')



    # show the plots
    # plt.show()