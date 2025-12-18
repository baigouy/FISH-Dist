import traceback
import numpy as np
from epyseg.deeplearning.deepl import EZDeepLearning
from batoolset.img import Img, save_as_tiff, _rotate_along_Z_axis, _recover_orig_after_rotation_along_z_axis, normalization
import os
import tempfile

def predict_single_image(deepTA, input_file, **predict_parameters):
    """
    Run prediction on a single image using a DeepTA model.

    This function prepares a temporary image file for compatibility with DeepTA's
    predict_generator API, generates predictions for the input image, and returns
    the results.

    Args:
        deepTA (object): An object implementing the following methods:
            - get_inputs_shape()
            - get_outputs_shape()
            - get_predict_generator(...)
            - predict_single(...)
        input_file (str or array-like): Path to the input image or an object
            convertible to an image by Img(...).
        **predict_parameters: Additional keyword arguments passed to both
            get_predict_generator() and predict_single().
            Special key:
                clip_by_frequency (optional): Passed only to get_predict_generator().

    Returns:
        any: Model prediction result(s) as returned by deepTA.predict_single().

    Example:
        ```python
        results = predict_single_image(deepTA_model, 'image.tif', clip_by_frequency=0.01)
        print(results)

        ```
    """


    input_shape = deepTA.get_inputs_shape()
    output_shape = deepTA.get_outputs_shape()

    # Store clip_by_frequency separately (suggested to rename local var to clip_freq)
    clip_by_frequency = predict_parameters.pop('clip_by_frequency', None)

    # Create a temporary image file for prediction generator
    temp = tempfile.NamedTemporaryFile(
        prefix="epyseg_tmp",
        suffix=".tif",
        delete=False
    )
    name = temp.name
    temp.close()

    # Save the input image into the temporary file
    # (Workaround for ImageJ saving issues).
    Img(input_file, dimensions='hwc').save(name)

    results = None

    try:
        # Build predict generator
        predict_generator = deepTA.get_predict_generator(
            inputs=[name],
            input_shape=input_shape,
            output_shape=output_shape,
            clip_by_frequency=clip_by_frequency,
            **predict_parameters  # passed to generator
        )

        final_predict_generator = predict_generator.predict_generator()

        # Run prediction
        results = deepTA.predict_single(
            final_predict_generator,
            output_shape,
            batch_size=1,
            **predict_parameters  # passed to predict_single
        )

    except Exception:
        traceback.print_exc()

    finally:
        # Always remove temporary file
        try:
            os.remove(name)
        except Exception:
            traceback.print_exc()

    return results

def predict_3D_stack_from_2D_model(
    deepTA,
    stack,
    apply_normalization_to_entire_stack_before=True,
    rotation_along_Z_axis_pattern=None,
    **predict_parameters
):
    """
    Predict a 3D image stack slice-by-slice using a 2D model.

    This function applies a 2D DeepTA model to each Z-slice of a 3D stack,
    optionally normalizing the stack globally, applying rotations along Z,
    and returning the combined predicted 3D stack.

    Args:
        deepTA (object): Model-like object implementing the same API as
            predict_single_image().
        stack (str or np.ndarray): Path to an image stack or array with shape
            (Z, H, W) or (Z, H, W, C). If a path is provided, the stack is
            loaded using Img().
        apply_normalization_to_entire_stack_before (bool, default=True):
            If True, normalize the entire stack globally before prediction.
        rotation_along_Z_axis_pattern (optional): If provided, rotates the stack
            along Z before prediction and un-rotates afterwards.
        **predict_parameters: Additional parameters passed to predict_single_image(),
            including optional "input_normalization" for normalization configuration.

    Returns:
        np.ndarray: Predicted 3D stack, shape (Z, H, W, ...) depending on model output.

    Example:
        ```python
        predicted_stack = predict_3D_stack_from_2D_model(deepTA_model, 'stack.tif', apply_normalization_to_entire_stack_before=True)

        ```
    """

    # --- Load stack if path was provided
    if isinstance(stack, str):
        stack = Img(stack).astype(float)

    # Ensure stack is Z, H, W, C for normalizer compatibility
    # Suggested variable name: ensure_stack_has_channel
    if len(stack.shape) == 3:
        stack = stack[..., np.newaxis]

    # --- Handle normalization
    input_norm_cfg = predict_parameters.get("input_normalization", None)

    if input_norm_cfg is not None and apply_normalization_to_entire_stack_before:
        print("normalizing stack")
        stack = normalization(stack, **input_norm_cfg)
    else:
        print("normalizing single images")

    # If global normalization was applied, disable per-image normalization below
    if apply_normalization_to_entire_stack_before:
        predict_parameters["input_normalization"] = None

    # --- Rotation (optional, can be useful especially if z is less than x, y but model may not work properly due to reduced resolution along Z)
    if rotation_along_Z_axis_pattern is not None:
        stack = _rotate_along_Z_axis(stack)

    final_output = []
    nb_z = stack.shape[0]

    # --- Slice-wise prediction using the 2D model
    for iii, img in enumerate(stack):
        print("current frame", iii, "/", nb_z)

        try:
            result = predict_single_image(deepTA, img, **predict_parameters)
        except Exception:
            print(f"Error predicting slice {iii}")
            traceback.print_exc()
            result = None

        if result is not None:
            result = np.squeeze(result)
        final_output.append(result)

    # Stack predictions back into a 3D array
    final_output = np.stack(final_output, axis=0)

    # --- Undo rotation if applied
    if rotation_along_Z_axis_pattern is not None:
        final_output = _recover_orig_after_rotation_along_z_axis(final_output)

    return final_output

if __name__ == '__main__':
    # TILE_WIDTH = 512
    # TILE_HEIGHT = 512
    # TILE_OVERLAP = 0

    TILE_WIDTH = 256
    TILE_HEIGHT = 256
    TILE_OVERLAP = 32

    input_channel_of_interest = None #None # 0 1 2
    input_normalization = None
    hq_predictions = 'mean'
    hq_pred_options = 'Only use pixel preserving augs (Recommended for CARE-like models/surface extraction)'
    input_normalization = {'method': 'Rescaling (min-max normalization)', 'range': [0, 1], 'individual_channels': True}

    predict_parameters = {}
    predict_parameters["input_channel_of_interest"] = input_channel_of_interest
    predict_parameters["default_input_tile_width"] = TILE_WIDTH
    predict_parameters["default_input_tile_height"] = TILE_HEIGHT
    predict_parameters["default_output_tile_width"] = TILE_WIDTH
    predict_parameters["default_output_tile_height"] = TILE_HEIGHT
    predict_parameters["tile_width_overlap"] = TILE_OVERLAP
    predict_parameters["tile_height_overlap"] = TILE_OVERLAP
    predict_parameters["hq_pred_options"] = hq_pred_options #"Use all augs (pixel preserving + deteriorating) (Recommended for segmentation)"  # not yet supported on current versions of epyseg
    predict_parameters["hq_predictions"] = hq_predictions #"Use all augs (pixel preserving + deteriorating) (Recommended for segmentation)"  # not yet supported on current versions of epyseg
    predict_parameters["post_process_algorithm"] = None  # 'Keep first channel only'# None
    predict_parameters["input_normalization"] = input_normalization # to be applied to the stack then reset to None for all other cases...

    # could loop over  
    # do a pipeline to shrink all the images
    # NB I have a small bug with images that smaller than size + TILE_overlap --> NEEDS BE FIXED
    deepTA = EZDeepLearning()

    # deepTA.load_or_build(model='/home/aigouy/mon_prog/Python/epyseg_pkg/personal/amrutha_gut/model_3_classes_small_images_only_125_epochs/linknet-vgg16-sigmoid-0.h5')
    # out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/Amrutha_gut/001000000000510622_dd75-fl_dd115-dig_21_for_splitted2.tif',apply_normalization_to_entire_stack_before=True, **predict_parameters)

    # print(out.shape)
    # save_as_tiff(out[...,0][...,np.newaxis], '/E/Sample_images/Amrutha_gut/crappy_test.tif') # dirty hack to keep the first channel only

    # I guess I may have a bug

    # NB THERE IS A BIG BUG --> NORMALIZATION IS NOT APPLIED PROPERLY BUT WHY IS THAT ??? --> BECAUSE I HAD FORGOTTEN TO SPECIFY THE CHANNEL OF INTEREST AND IMAGE OF MANUE WAS MULTI CHANNEL --> ALL IS OK IN FACT

    #DO SPECIFY CHANNEL OF INTEREST IF MULTI CHANNELS

    # test spot detection
    # predict_parameters["input_channel_of_interest"] = 1
    # deepTA.load_or_build(model='/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/220922_first_test_not_bad/linknet-vgg16-sigmoid-0.h5')

    # test nuclear detection
    predict_parameters["input_channel_of_interest"] = 0
    # for vertical need redefine the stuff --> may not work for nuclei --> need take the multiple of 8 that is closest to the new height --> TODO
    # TILE_WIDTH = 256
    # TILE_HEIGHT = 8
    # TILE_OVERLAP = 0
    # deepTA.load_or_build(model='/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/220930_first_test_nuc_based_on_cellpose_diameter_30/linknet-vgg16-sigmoid-0.h5')
    deepTA.load_or_build(model='/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/221012_second_test_training_spots_with_gradients_and_imroved_ellipse_GT/linknet-vgg16-sigmoid-0.h5')
    # out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/X1 633 X2 565 yx1 yx2 m 12ng ul probes Hoe 1 2000 3 post.tif',apply_normalization_to_entire_stack_before=True, **predict_parameters)
    # deepTA.load_or_build(model='/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/trained_models/220919_test_detection_nuclei_directly_from_spot_channels/linknet-vgg16-sigmoid-0.h5')
    # out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-1_ch1.tif',apply_normalization_to_entire_stack_before=True,rotation_along_Z_axis_pattern=True, **predict_parameters)
    out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-1_ch1.tif',apply_normalization_to_entire_stack_before=True, **predict_parameters)
    # out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-2_ch2.tif',apply_normalization_to_entire_stack_before=True, rotation_along_Z_axis_pattern=True, **predict_parameters)
    # out = predict_3D_stack_from_2D_model(deepTA,'/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/X1 633 X2 565 yx1 yx2 m 12ng ul probes Hoe 1 2000 3 post-ch2_only.tif',apply_normalization_to_entire_stack_before=False, **predict_parameters)


    # I could also try predict a single time on max proj in this dimension when rotated along the Z axis because otherwise it takes forever...

    print(out.shape)
    # force manue output to be dhwc
    # save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/X1 633 X2 565 yx1 yx2 m 12ng ul probes Hoe 1 2000 3 nuc detection.tif') # dirty hack to keep the first channel only
    # save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-1_ch1 nuc detection.tif') # dirty hack to keep the first channel only
    # save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-2_ch2 nuc detection.tif') # dirty hack to keep the first channel only
    # save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-2_ch2 nuc detection_perp.tif') # dirty hack to keep the first channel only
    # save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-1_ch1_perp.tif') # dirty hack to keep the first channel only
    save_as_tiff(out[...,np.newaxis], '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/double_points_coloc_different_acquisition_speed_n_pixel_size/X2 633 X2 565 CantonS f NS3a-1_ch1_no_rot.tif') # dirty hack to keep the first channel only


