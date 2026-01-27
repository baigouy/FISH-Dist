# --root-path "/media/teamPrudhomme/EqpPrudhomme2/FISH_Dist_analysis"
# --config /media/teamPrudhomme/EqpPrudhomme2/FISH_Dist_analysis/config.json

from batoolset.files.tools import smart_name_parser
from batoolset.utils.loadlist import loadlist
from fishdist.automated_analysis_pipeline import print_folder_logic_with_values, parse_arguments, \
    merge_args_with_config, validate_paths, print_current_analysis, move_to_DONE_when_DONE
import os
import sys
from fishdist.fish_analysis_pipeline import run_analysis

# from epyseg.epygui import EPySeg
# def run_FISH_dist():

if __name__ == '__main__':
    print_folder_logic_with_values()
    args = parse_arguments()
    args = merge_args_with_config(args)
    validate_paths(args)

    # Prepare folders_to_scan based on the arguments
    if hasattr(args, 'root_path'):
        folders_to_scan = [
            os.path.join(args.root_path, 'colocs'),
            os.path.join(args.root_path, 'distances')
        ]
        controls_path = os.path.join(args.root_path, 'controls')
    else:
        folders_to_scan = [args.colocs_path, args.distances_path]
        controls_path = args.controls_path

    print_current_analysis(
        nucleus_channel=args.nucleus_channel,
        reference_channel=args.reference_channel,
        folders=folders_to_scan,
        controls_path=controls_path,
        run_gaussian_fit=args.run_gaussian_fit,
        nuclear_model_to_use=args.nuclear_model_to_use,
        spot_model_to_use=args.spot_model_to_use
    )

    print('folders_to_scan, controls_path', folders_to_scan, controls_path)

    # Print configuration
    print("\nAnalysis Configuration:")
    print(f"- Nucleus channel: {args.nucleus_channel}")
    print(f"- Reference FISH channel: {args.reference_channel}")
    print(f"- Second spot channel(s): {args.second_spot_channel}")
    print(f"- Pairing threshold: {args.pairing_threshold} nm")
    print(f"- Area threshold: {args.area_threshold}")
    print(f"- Run segmentation: {args.run_seg}")
    print(f"- Run distance measurements: {args.run_distance_measurements}")
    print(f"- Run controls: {args.run_ctrls}")
    print(f"- Run Gaussian fit: {args.run_gaussian_fit}")
    print(f"- Nuclear model: {args.nuclear_model_to_use}")
    print(f"- Spot model: {args.spot_model_to_use}")

    # Prepare paths for analysis
    if hasattr(args, 'root_path'):
        folders_to_scan = [
            os.path.join(args.root_path, 'colocs'),
            os.path.join(args.root_path, 'distances')
        ]
    else:
        folders_to_scan = [args.colocs_path, args.distances_path]

    first_spot_channel = args.reference_channel
    second_spot_channel = args.second_spot_channel

    if not folders_to_scan:
        print("No valid folders provided. Exiting.")
        sys.exit(0)

    for folder in folders_to_scan:
        paths = loadlist(folder + '/*.czi')+loadlist(folder + '/*.tif') # TODO support for czi or tifs, shall I add more ? maybe ok for now
        if not paths:
            print('No images found --> nothing to do')
            continue

        list_pairs_for_reg = loadlist(controls_path + '/*.czi')

        RUN_REG = False
        if 'colocs' in folder:
            list_pairs_for_reg = paths
            print(('PATH USED FOR REGISTRATION OVERRIDDEN', list_pairs_for_reg))
            RUN_REG = True

        print('paths', paths)

        PAIRING_THRESHOLD = args.pairing_threshold
        if list_pairs_for_reg:
            correction_matrix_save_path = smart_name_parser(list_pairs_for_reg[0],
                                                            'parent')  # TODO is this stuff really needed -−> I guess not
        else:
            correction_matrix_save_path = controls_path
        RUN_SEG = args.run_seg
        RUN_DISTANCE_MEASUREMENTS = args.run_distance_measurements
        RUN_CTRLS = args.run_ctrls
        RUN_GAUSSIAN_FIT = args.run_gaussian_fit
        area_threshold = args.area_threshold

        run_analysis(
            paths,
            correction_matrix_save_path=correction_matrix_save_path,
            PAIRING_THRESHOLD=PAIRING_THRESHOLD,
            area_threshold=area_threshold,
            RUN_SEG=RUN_SEG,
            RUN_REG=RUN_REG,
            RUN_DISTANCE_MEASUREMENTS=RUN_DISTANCE_MEASUREMENTS,
            RUN_CTRLS=RUN_CTRLS,
            RUN_GAUSSIAN_FIT=RUN_GAUSSIAN_FIT,
            list_pairs_for_reg=list_pairs_for_reg,
            first_spot_channel=first_spot_channel,
            second_spot_channel=second_spot_channel,
            nuclear_model_to_use=args.nuclear_model_to_use,
            spot_model_to_use=args.spot_model_to_use
        )

        move_to_DONE_when_DONE(paths, add_extras=True)