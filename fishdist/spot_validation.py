import numpy as np
from skimage.measure import label, regionprops

from batoolset.img import Img, dilate_3D_spots_a_bit
from batoolset.nps.tools import convert_numpy_bbox_to_coord_pairs


def validate_spots(spots, discard_spot_size_rule=None, dilate_spot_size_rule=None,
                   intensity_image_for_regions_optional=None):
    """
    Validate and optionally dilate 3D spot ROIs based on size criteria.

    This function labels connected components in a 3D spot image, computes region
    properties, discards small spots, and optionally dilates spots below a size threshold.
    Returns a list of validated (and potentially dilated) ROIs.

    Args:
        spots (np.ndarray): Binary or probabilistic 3D image containing candidate spots, shape (Z, Y, X).
        discard_spot_size_rule (int, optional): Minimum size in any dimension below which a spot is discarded. Default is None.
        dilate_spot_size_rule (int, optional): Minimum size in any dimension below which a spot is dilated. Default is None.
        intensity_image_for_regions_optional (np.ndarray, optional): Optional intensity image for computing region properties. Default is None.

    Returns:
        list of skimage.measure._regionprops.RegionProperties: List of validated and optionally dilated ROIs.

    Example:
        ```python
        import numpy as np
        from skimage.measure import regionprops
        spots = np.random.rand(10, 20, 20) > 0.95
        valid_rois = validate_spots(spots, discard_spot_size_rule=2, dilate_spot_size_rule=4)
        print(len(valid_rois))
        ```
    """

    __PREVIEW__ = False

    # Label connected components in the binary spot image
    lab_spots = label(spots > 0.5, connectivity=None, background=0)

    # Compute initial region properties
    if intensity_image_for_regions_optional is None:
        rps = regionprops(lab_spots)
    else:
        rps = regionprops(lab_spots, intensity_image=intensity_image_for_regions_optional)

    # If no size rules provided, return all regions
    if discard_spot_size_rule is None and dilate_spot_size_rule is None:
        return rps

    # Containers for ROIs to dilate and validated ROIs
    ROIs_to_be_dilated = []
    valid_ROIS = []

    # Process each region
    for region in rps:
        bbox = convert_numpy_bbox_to_coord_pairs(region.bbox)  # Convert bounding box to min/max coordinates
        dimension_size = bbox[:, 1] - bbox[:, 0]  # Size in each dimension

        if __PREVIEW__:
            print('dimension_size', dimension_size)

        # Discard spots smaller than discard_spot_size_rule
        if discard_spot_size_rule is not None and any(dimension_size <= discard_spot_size_rule):
            if __PREVIEW__:
                print('ROI too small; discarded')
            continue

        # Mark spots smaller than dilate_spot_size_rule for dilation
        if dilate_spot_size_rule is not None and any(dimension_size < dilate_spot_size_rule):
            ROIs_to_be_dilated.append(region)
            if __PREVIEW__:
                print('ROI too small; marked for dilation')
        else:
            if __PREVIEW__:
                print('Valid ROI')
            valid_ROIS.append(region)

    # Dilate marked ROIs if any
    if ROIs_to_be_dilated:
        dilated_spots = dilate_3D_spots_a_bit(spots > 0.5)
        lab_spots_dilated = label(dilated_spots, connectivity=None, background=0)

        if intensity_image_for_regions_optional is None:
            rps_dilated = regionprops(lab_spots_dilated)
        else:
            rps_dilated = regionprops(lab_spots_dilated, intensity_image=intensity_image_for_regions_optional)

        # Find corresponding dilated ROIs
        dilated_ROIs = find_corresponding_dilated_ROIs(ROIs_to_be_dilated, lab_spots_dilated, rps_dilated=rps_dilated)

        # Add dilated ROIs to validated ROIs
        valid_ROIS.extend(dilated_ROIs)

    return valid_ROIS


def find_corresponding_dilated_ROIs(rps_to_search_for, lab_dilated, rps_dilated=None):
    """
    Identify the corresponding dilated ROIs for a list of original regions.

    For each region in `rps_to_search_for`, this function finds the matching region
    in a dilated labeled image (`lab_dilated`). Optionally, it can return the
    matching RegionProperties if `rps_dilated` is provided.

    Args:
        rps_to_search_for (list of skimage.measure._regionprops.RegionProperties):
            List of original ROIs to find in the dilated image.
        lab_dilated (np.ndarray): Labeled 3D array representing the dilated image.
        rps_dilated (list of skimage.measure._regionprops.RegionProperties, optional):
            Optional list of RegionProperties for the dilated image. If provided,
            the function returns the matching RegionProperties objects instead of raw coordinates. Default is None.

    Returns:
        list of np.ndarray or skimage.measure._regionprops.RegionProperties or None:
            List of coordinates (np.ndarray) or RegionProperties corresponding to the
            input regions. If a region cannot be found, None is returned in that position.

    Example:
        ```python
        from skimage.measure import label, regionprops
        import numpy as np

        spots = np.random.rand(5, 10, 10) > 0.9
        lab = label(spots)
        rps = regionprops(lab)

        # simulate dilation
        spots_dilated = np.copy(spots)
        lab_dilated = label(spots_dilated)
        extended_rois = find_corresponding_dilated_ROIs(rps, lab_dilated)
        print(len(extended_rois))
        ```
    """
    extended_ROIS = []
    for region in rps_to_search_for:
        cell_ID = lab_dilated[tuple(region.coords[0].T)]  # get the first point color
        if cell_ID == 0:
            # cell not found --> ignoring
            extended_ROIS.append(None)
            continue

        if rps_dilated is not None:
            region_found = False
            for region2 in rps_dilated:
                if cell_ID == region2.label:
                    region_found = True
                    extended_ROIS.append(region2)
                    break
            if region_found:
                continue

        # I need get the coords
        coords = np.where(lab_dilated == cell_ID)
        coords = tuple(zip(*coords))
        coords = np.asarray(coords)

        extended_ROIS.append(coords)

    return extended_ROIS


if __name__ == '__main__':

    # if True:
    #     test_dim = np.asarray([1,1,3])
    #     test_threshold_dim_size = np.asarray([1,1,3])
    #
    #     if all(test_dim>=test_threshold_dim_size):
    #         print('ok')
    #     else:
    #         print('false')
    #
    #     test_dim = np.asarray([1, 1, 1])
    #     if all(test_dim>=test_threshold_dim_size):
    #         print('ok')
    #     else:
    #         print('false')
    #
    #     import sys
    #     sys.exit(0)


    spots = Img(
        '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/sensitivity_issue/X1 633 X2 565 yx1 yx2 m 12ng ul probes Hoe 1 2000 2/ch1.tif')
    # print(validate_spots(spots,discard_spot_size_rule=[2,1,1] )) # removes all spots that are not extending over at least 2 Z frames
    # print(validate_spots(spots,discard_spot_size_rule=[0,1,1] )) # deletes all single points # --> pb deletes also 1 2 1 --> how can I change that --> mayeb ok or need recode if I want it to be smarter
    print(validate_spots(spots,discard_spot_size_rule=[0,1,1], dilate_spot_size_rule=[2,0,0] )) # deletes all single points and dilate the points that have a too small Z dimension # --> pb deletes also 1 2 1 --> how can I change that --> mayeb ok or need recode if I want it to be smarter


    #shall I use any