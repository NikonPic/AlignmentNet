# Comprehensive Lower Extremity Alignment Analysis Incorporating Multi-Scale and Multi-Task Deep Learning
>This repository represents the source code for a completely automated alignment analysis software


 <img src="architecture.PNG" alt="Drawing" style="width: 1200px;">


## Setup

* Install Python (Recommended 3.6+)
* Pytorch (Recommended 1.7+)
* Detectron2 (Recommended 0.2+)

## What does each file do? 

    .     
    ├── src                              # Source Code
    │   ├── main.py                      # main function, to preprocess, train and evaluate
    │   ├── helpers.py                   # helpers for preprocessing the data to COCO
    │   ├── categories.py                # def of all individual categories to split
    │   ├── train_detectron.py           # training and managing of all submodules
    │   ├── optimize_image_range.py      # Local opt. for maximum img accuracy
    │   ├── nuber_detection.py           # detect ruler if no sphere is available
    │   ├── angle_calc.py                # calculate the alignmant angles
    │   ├── angles_vis.py                # visualize aliognment angles
    │   ├── eval_angle_test.py           # perform all evaluations
    │   └── extern_studies.py            # helpers for external analysis
    |
    ├── jsons                            # Folder containing all Networks
    │   └── categories                   # Subfolder with the specialised networks and sub-train datasets
    |
    ├── images                           # Folder with all training images
    |
    └── results                          # Contains the final results

# Citation

If you use this project in any of your work, please cite:

```
tbd.
```