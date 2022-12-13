#
#  main.py
#  Training and Evaluation
#
#  Created by Nikolas Wilhelm on 2021-02-19.
#  Copyright © 2021 Nikolas Wilhelm. All rights reserved.
#


# %% Imports
from extern_studies import FILES_EXT, preview_external, update_ext
import helpers as hp
from train_detectron import train_all_subnetworks, train_main_detector, Evaluator
from eval_angle_test import perform_all_tests
from categories import CATNAMES
from ipywidgets import widgets


# %% Training and Internal

if __name__ == '__main__':
    
    train = False
    remake_data = False
    train_networks = False
    intra_mode = False

    # add the source path here
    paths = [
        'D:\data\Osteosyn4\Testdaten',
        'D:\data\Osteosyn4\Trainingsdaten',
    ]

    if train:

        if remake_data:
            patients = hp.create_image_repo(paths, exclude=False)
            hp.create_cocos(hp.SPLIT, from_excel_test=True)
            hp.distribute_jsons_to_single_class(hp.SPLIT.keys())

        if intra_mode:
            patients = hp.create_image_repo(
                paths, exclude=False, img_path='new_images')
            hp.create_cocos(hp.SPLIT, from_excel_test=True, intramode=True)

        if train_networks:
            train_main_detector(True)
            train_all_subnetworks(True, CATNAMES)

        perform_all_tests(mode='test')
        preview_external()

    # live evaluation 1
    evaluator = Evaluator(onlyimage=True, from_eval_model=True, draw_annotations=True,
                          extra_analysis=True, angle_vis=False, ocr_analysis=False, save=True)
    idx = widgets.IntSlider(25, min=0, max=177)
    widgets.interact(evaluator.__call__, idx=idx, optimize_area=True)

# %%