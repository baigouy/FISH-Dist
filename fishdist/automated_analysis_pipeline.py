# DO I REALLY NEED TO MAKE THIS CODE PUBLIC ???? --> I GUESS NOT

import sys

# sys.path.append('/home/aigouy/miniconda3/pkgs/cuda-nvcc-tools-12.4.131-h99ab3db_0/bin')
# sys.path.append('/home/aigouy/miniconda3/bin')
# sys.path.append('/home/aigouy/miniconda3/condabin')

import os
import tensorflow as tf

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

print('entered the script')
print('__file__',__file__)

import os.path
import traceback
from batoolset.files.tools import smart_name_parser
from batoolset.utils.loadlist import loadlist
from fishdist.fish_analysis_pipeline import \
    run_analysis
import shutil

print('really executing the script')

def move_to_DONE_when_DONE(lst, add_extras=False):
    """
    Move a list of files (and optionally associated extra files) into a 'DONE' subdirectory.

    Args:
        lst (list of str): List of file paths to move.
        add_extras (bool, default=False): If True, also move additional files in the same parent
            directory with extensions .png, .npy, or .pdf.

    Returns:
        None: Files are moved to a 'DONE' folder; the function does not return a value.

    Example:
        ```python
        move_to_DONE_when_DONE(['/path/to/file1.txt', '/path/to/file2.txt'], add_extras=True)

        ```
    """

    extras = []
    if add_extras:
        # add the extra reg files
        extras = loadlist(smart_name_parser(lst[0], 'parent')+'/*.png')
        extras += loadlist(smart_name_parser(lst[0], 'parent')+'/*.npy')
        extras += loadlist(smart_name_parser(lst[0], 'parent')+'/*.pdf')

    extras.extend(lst)

    for file in extras:
        filename = smart_name_parser(file,'short')
        parent = smart_name_parser(file, 'parent')
        associated_folder = smart_name_parser(file,'short_no_ext')
        dest = parent+'/DONE'
        os.makedirs(dest, exist_ok=True)

        try:
            shutil.move(file, os.path.join(dest,filename))
            try:
                if os.path.exists(smart_name_parser(file,'full_no_ext')):
                    shutil.move(smart_name_parser(file,'full_no_ext'), dest)
            except:
                print('no output folder --> skipping')
        except:
            traceback.print_exc()
            # print error

if __name__ == '__main__':

    if True:
        # probably useless but keep ---> CAN ADD sudo /home/aigouy/remount_samba.sh as an application au démarrage et si ajouté au sudoers alors ça peut marcher --> cool!!!
        if not os.path.exists('/media/teamPrudhomme/EqpPrudhomme2/'):
            try:
                os.system('sudo /home/aigouy/remount_samba.sh')  # opens and closes internet connection --> just a bug fix to be sure everything is ok
            except:
                traceback.print_exc()

        if not os.path.exists('/media/teamPrudhomme/EqpPrudhomme2/'):
            print('Could not connect to the fileserver --> quitting')
            sys.exit(0)


    # folders_to_scan = ['/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/3072', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/3072']
    folders_to_scan = ['/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/3072', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/3072']
    # folders_to_scan = ['/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/tst_benoit_2048','/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/3072', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/2048', '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/colocs/3072']
    # folders_to_scan = ['/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/3072']

    for folder in folders_to_scan:
        # use files in a specific folder for registration
        if folder.replace('/','').endswith('2048'):
            # please stick to that from now on
            # or allow for tiff support ????
            # UNFORTUNATELY MANUE CHANGED AGAIN THE ORDER OF THE CHANNELS WHICH CREATES HUGE COMPAT PROBLEMS --> AND IS LIKELY TO CAUSE UNPRECENTED ISSUES
            first_spot_channel = -2 # DIRTY FIX FOR MANUE CHANGE
            second_spot_channel = -1
            DEFAULT_PATH_2048='/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Coloc X2 2048/*.czi'
            list_pairs_for_reg = loadlist(DEFAULT_PATH_2048)  # for 2048x2048 images
            NEW_PATH = '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/controls/2048/*.czi'
            controls = loadlist(NEW_PATH)
            if controls: # if there are files in the controls -> use them for registration, if not use the defaut files
                list_pairs_for_reg=controls
                print(('PATH USED FOR REGISTRATION', NEW_PATH))
            else:
                print(('PATH USED FOR REGISTRATION', DEFAULT_PATH_2048))
        else:
            # UNFORTUNATELY MANUE CHANGED AGAIN THE ORDER OF THE CHANNELS WHICH CREATES HUGE COMPAT PROBLEMS --> AND IS LIKELY TO CAUSE UNPRECENTED ISSUES
            first_spot_channel = 1
            second_spot_channel = -1 # order of the channels in /E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit R1 R6

            DEFAULT_PATH_3072 = '/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/full_real_analysis_tst1/Benoit colocalisation on pupal wingsColocalisation_not_blurred/*.czi'
            list_pairs_for_reg = loadlist(DEFAULT_PATH_3072) # for 3072 px images
            NEW_PATH = '/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/controls/3072/*.czi'
            controls = loadlist(NEW_PATH)
            if controls: # if there are files in the controls -> use them for registration
                list_pairs_for_reg=controls
                print(('PATH USED FOR REGISTRATION', NEW_PATH))
            else:
                print(('PATH USED FOR REGISTRATION',DEFAULT_PATH_3072))

        paths = loadlist(folder+'/*.czi')

        if not paths:
            print('no images found --> nothing todo')
            # import sys
            # sys.exit(0)
            continue

        # get the logic and finalize
        RUN_REG = False
        if 'colocs' in folder:
            list_pairs_for_reg = paths
            print(('PATH USED FOR REGISTRATION OVERRIDDEN', list_pairs_for_reg))
            RUN_REG = True

        print('paths', paths)

        PAIRING_THRESHOLD = 30 # 30 # 60 # --> 4µm 30 #--> 2µm # 250 #--> initial size   # tested for Sarah 60 30 15
        correction_matrix_save_path = smart_name_parser(list_pairs_for_reg[0], 'parent')

        area_threshold = None # 35 for sarah (before) # None for Manue
        RUN_SEG = True
        RUN_DISTANCE_MEASUREMENTS = True
        RUN_CTRLS = True

        run_analysis(paths, correction_matrix_save_path=correction_matrix_save_path,
                     PAIRING_THRESHOLD=PAIRING_THRESHOLD, area_threshold=area_threshold, RUN_SEG=RUN_SEG,
                     RUN_REG=RUN_REG, RUN_DISTANCE_MEASUREMENTS=RUN_DISTANCE_MEASUREMENTS, RUN_CTRLS=RUN_CTRLS,list_pairs_for_reg=list_pairs_for_reg,
                     first_spot_channel = first_spot_channel,second_spot_channel = second_spot_channel # dirty bug fix for manue change of order of channels (need be extremely careful from now on...)
                     )

        move_to_DONE_when_DONE(paths, add_extras=True)

