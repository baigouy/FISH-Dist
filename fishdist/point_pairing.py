from scipy.spatial.distance import cdist
import traceback
import numpy as np
from skimage.filters import difference_of_gaussians
# from personal.cellpose.cell_pose_me_2D_seg_just_return_seg import segment_nucleus
from skimage.measure import label, regionprops


def pair_points_by_distance(points1, points2, threshold=3, rescaling_factor=None):
    """
    Pair points from two sets based on minimal Euclidean distance within a threshold.

    Each point in `points1` is matched to the closest point in `points2` if the
    distance is below `threshold`. Existing pairings are updated if a closer match
    is found.

    Args:
        points1 (np.ndarray of shape (N, 3)): First set of 3D points (e.g., measured points).
        points2 (np.ndarray of shape (M, 3)): Second set of 3D points (e.g., reference points).
        threshold (float or None, default=3): Maximum allowed distance for pairing.
        rescaling_factor (array-like of length 3 or None, default=None): Scaling factor to convert
            points into physical units (e.g., voxel size).

    Returns:
        dict: Mapping of points from `points1` to their closest point in `points2`.

    """

    point_pairs = {}
    point_pairs_distances = {}

    try:
        # Check for empty arrays
        if isinstance(points1, np.ndarray) and points1.size == 0:
            print('points1 is empty; pairing impossible')
            return point_pairs
        if isinstance(points2, np.ndarray) and points2.size == 0:
            print('points2 is empty; pairing impossible')
            return point_pairs

        # Apply optional rescaling
        if rescaling_factor is not None:
            rescaling_factor = np.asarray(rescaling_factor)
            points1*=rescaling_factor
            points2*=rescaling_factor

        # Compute pairwise Euclidean distances
        pairwise_distances = cdist(points1, points2, metric='euclidean')

        for idx, distances in enumerate(pairwise_distances):
            current_distance = distances.min()

            # Skip if distance exceeds threshold
            if threshold is not None and current_distance > threshold:
                continue

            pt1 = tuple(points1[idx].tolist())
            pt2 = tuple(points2[np.argmin(distances)].tolist())

            # Skip if pt1 already paired to a closer point
            if pt1 in point_pairs:
                if point_pairs_distances[pt1] < current_distance:
                    continue

            # Check if pt2 is already paired
            if pt2 in point_pairs.values():
                existing_key = _get_key_by_val(point_pairs, pt2)
                if existing_key is not None:
                    if point_pairs_distances[existing_key] < current_distance:
                        continue
                    else:
                        # Replace previous pairing with the closer one
                        point_pairs.pop(existing_key, None)
                        point_pairs_distances.pop(existing_key, None)

            # Add or update pairing
            point_pairs[pt1] = pt2
            point_pairs_distances[pt1] = current_distance

    except:
        traceback.print_exc()

    return point_pairs


def _get_key_by_val(my_dict, val):
    """
    Find the first key in a dictionary that maps to a specified value.

    Args:
        my_dict (dict): Dictionary to search.
        val (any): Value to locate in the dictionary.

    Returns:
        any or None: The key corresponding to the given value, or None if not found.

    Example:
        >>> my_dict = {'a': 1, 'b': 2, 'c': 3}
        >>> key = _get_key_by_val(my_dict, 2)
        >>> print(key)
        b
    """

    for key, value in my_dict.items():
        if val == value:
            return key
    return None






# TODO trash ?????

#
# def cellpose_2D_seg_mask(nuclear_channel_3D, diameter=None):
#     # do max and avg proj
#
#     print(nuclear_channel_3D.shape)
#
#     mx_proj = max_proj(nuclear_channel_3D)
#
#
#     # plt.imshow(mx_proj)
#     # plt.show()
#     # print(mx_proj.shape, mx_proj.dtype)
#
#
#     # required for proper preview by matplotlib
#     # plt.imshow(max_proj/max_proj.max())
#     # plt.show()
#
#     av_proj = avg_proj(nuclear_channel_3D) # this makes binarization not so bad even with the crappy bg she has --> see how cellpose performs on that ???
#
#     # plt.imshow(av_proj/av_proj.max())
#     # plt.show()
#
#     # pass
#
#     # now segment both images with cellpose and get the masks and combine them
#
#     # then save this
#
#     # tst = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue/AVG_Canton S m X1 633 X2 565-1.png'
#     nuc_max = segment_nucleus(mx_proj, diameter=diameter)
#     nuc_avg = segment_nucleus(av_proj, diameter=diameter)
#
#     # plt.imshow(nuc_max)
#     # plt.show()
#
#     # plt.imshow(nuc_max)
#     # plt.show()
#     #
#     # plt.imshow(nuc_avg)
#     # plt.show()
#
#     # plt.imshow(nuc)
#     # plt.show()
#
#     # find overlapping objects and try to combine them
#     # TODO
#     # find rules for combination --> such as and or not and whether I should put the intersection or not or whether there should be rules for fusion
#     # in fact I can recycle the stuff I did for Laurence and even further improve it !!!
#
#     mask = segmentation_mask_fuser(list_of_masks_to_fuse=[nuc_max, nuc_avg])
#
#
#     # try save all three masks just to see what it gives
#
#     # Img(nuc_max,dimensions='hw').save('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue/220325 X2a565 X2a633_coloc/mask_max.tif')
#     # Img(nuc_avg, dimensions='hw').save('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue/220325 X2a565 X2a633_coloc/mask_avg.tif')
#     # Img(mask, dimensions='hw').save('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue/220325 X2a565 X2a633_coloc/mask_zfused.tif')
#
#
#     return mask
#
# # fuses all masks with a set of rules


def apply_nuclear_mask(loci_dots, nuclear_mask):
    # print(loci_dots.shape, nuclear_mask.shape)
    if nuclear_mask is not None:
        if len(loci_dots.shape) != len(nuclear_mask.shape):
            # reset anything outside the mask
            for zzz, img in enumerate(loci_dots):
                img[nuclear_mask == 0] = 0
                loci_dots[zzz] = img
        else:
            loci_dots[nuclear_mask == 0] = 0
    return loci_dots



# could also have and or not
# TODO --> really implement the logic for fusion
fusion_methods = ['biggest', 'smallest', 'fusion', 'and', 'or', 'not']
def segmentation_mask_fuser(list_of_masks_to_fuse=None, fusion_method='biggest'):
    if list_of_masks_to_fuse is None or not list_of_masks_to_fuse:
        return None
    if len(list_of_masks_to_fuse)==1:
        # only one image --> nothing to do
        return list_of_masks_to_fuse[0]
    # checks made --> now proceed to fusion
    fused_mask = np.copy(list_of_masks_to_fuse[0])

    # get all cells of both images and find matching pairs
    # I need label the cells except if they are already labeled --> in fact assume they are already labeled !!!

    # in fact I need to do that for all the cells and I need to keep this for all the cells or just for the fused image maybe
    # I need increment id in a different manner also --> see how to do that


    # now loop over all other cells and depending on the rule do things
    for seg in list_of_masks_to_fuse[1:]:
        rps = regionprops(seg)
        # check if any overlap and take action
        for region in rps:
            # count cells in mask below and take action depending on the rules
            ids_cells_below, count = np.unique(fused_mask[region.coords[:, 0], region.coords[:, 1]],return_counts=True)
            bckup_ids_cells_below = ids_cells_below
            bckup_count = count

            ids_cells_below = ids_cells_below.tolist()
            count = count.tolist()

            for zzz, id in enumerate(ids_cells_below):
                if id == 0:
                    continue
                # if a cell is engulfed in a bigger one then make sure to exclude it
                # if count[zzz] >= 50 / 100 * rps[int(id - 1)].area:
                #     detected.append(rps[int(id - 1)])

            # if only 0 below --> add the cell --> this is and mode


            count_sort_ind = np.argsort(- bckup_count)
            most_frequent = bckup_ids_cells_below[count_sort_ind][0]
            if most_frequent == 0:
                try:
                    most_frequent = bckup_ids_cells_below[count_sort_ind][1]
                except:
                    continue
            # do something with the most frequent cell

            if most_frequent == 0:
                # add the new cell to it with a new ID
                fused_mask[region.coords[:, 0], region.coords[:, 1]]=fused_mask.max()+1

            if most_frequent != 0:
                fused_mask[region.coords[:, 0], region.coords[:, 1]] = most_frequent







    # assume 2D images only for simplicity but maybe go 3D some day
    # for the first image I can copy it directly to





    return fused_mask



