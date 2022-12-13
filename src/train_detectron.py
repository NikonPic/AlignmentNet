# %%
from typing import overload
import cv2
import numpy as np
import torch
import json
import os
import shutil
from tqdm import tqdm
import pandas as pd
from detectron2.engine.defaults import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from categories import CATNAMES
from PIL import Image, ImageDraw, ImageFont
from imantics import Mask
import matplotlib.pyplot as plt

# personal functions
from angle_vis import draw_all_angles
from helpers import create_local_directory
from number_detection import ocr_analysis
from optimize_image_range import get_img_cut_dims

MODEL_NAME = 'model_final.pth'
MODEL_BACKBONE = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"


def t1filt(arr, t=0.8):
    """filter the array with threshold t"""
    arr_new = []
    arr_new.append(arr[0])
    loc_value = arr[0]

    for i in range(1, len(arr)):
        loc_value = t * loc_value + (1 - t) * arr[i]
        arr_new.append(loc_value)

    for i in range(len(arr)-1, 0, -1):
        loc_value = t * loc_value + (1 - t) * arr_new[i]
        arr_new[i] = loc_value

    return arr_new


def kantenfilt(arr):
    new_arr = []
    new_arr.append(0)

    for i in range(1, len(arr)-1):
        loc_value = 0.5 * (-arr[i-1] + arr[i+1])
        new_arr.append(loc_value)

    new_arr.append(0)
    return new_arr


def draw_bbox(anno, draw, fill=(0, 255, 0, 150), w=2):
    if 'bbox' in anno.keys() and (len(anno['bbox']) > 2):
        x, y, width, height = anno['bbox']
        draw.line((x, y, x+width, y), fill=fill, width=w)
        draw.line((x+width, y, x+width, y + height), fill=fill, width=w)
        draw.line((x+width, y+height, x, y+height), fill=fill, width=w)
        draw.line((x, y+height, x, y), fill=fill, width=w)


def draw_keypoints(cat_infos, anno, draw, radius=5, fill=(0, 255, 0, 255)):
    if 'keypoints' in anno.keys():
        keypoint_arr = anno['keypoints']
        if len(keypoint_arr) > 2:
            x_arr, y_arr = keypoint_arr[::3], keypoint_arr[1::3]

            try:
                font = ImageFont.truetype("../fonts/arial.ttf", 30)
            except:
                font = None

            for i, (x, y) in enumerate(zip(x_arr, y_arr)):
                keypoint_name = cat_infos[0]["keypoints"][i]
                draw.ellipse(
                    (x-radius, y-radius, x+radius, y+radius), fill=fill)
                draw.text((x, y), keypoint_name, fill=fill,
                          align='left', font=font)


def draw_segmentations(anno, draw, fill=(0, 255, 0, 20)):
    if 'segmentation' in anno.keys():
        segmentation = anno['segmentation']
        out_fill = list(fill).copy()
        out_fill[-1] = 255
        out_fill = tuple(out_fill)
        for poly in segmentation:
            if len(poly) > 2:
                draw.polygon((poly), fill, outline=out_fill)


def draw_labels(cat_infos, anno, draw, mask):
    draw_keypoints(cat_infos, anno, draw)
    draw_bbox(anno, draw)
    draw_segmentations(anno, draw) if mask else None


def draw_network_labels(cat_infos, anno, draw, mask):
    draw_keypoints(cat_infos, anno, draw, fill=(50, 50, 255, 200))
    draw_bbox(anno, draw, fill=(50, 50, 255, 75))
    draw_segmentations(anno, draw, fill=(50, 50, 255, 20)) if mask else None


def update(path, predictor, idx=10, truelabel=True, mask=True, mode='test'):
    """perform visualization on the images"""
    imgpath = f'{path}/images'
    jsonpath = f'{path}/{mode}.json'

    with open(jsonpath) as jf:
        data = json.load(jf)

    cat_infos = data["categories"]
    img_infos = data['images']
    img_id_map = {}
    for i, img_info in enumerate(img_infos):
        img_id_map[img_info["id"]] = i

    anno = data['annotations'][idx]
    img_id = anno['image_id']
    img_name = img_infos[img_id_map[img_id]]['file_name']
    img_name = f'{imgpath}/{img_name}'

    img = cv2.imread(img_name)

    with torch.no_grad():
        outputs = predictor(img)

    instances = outputs["instances"].to("cpu")[:1]
    anno_net = instance2anno(instances, anno)

    img = Image.fromarray(img, 'RGB')
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw the true label
    if truelabel:
        draw_labels(cat_infos, anno, draw, mask)

    # draw the labels of the network
    draw_network_labels(cat_infos, anno_net, draw, mask)

    return img


def instance2anno(instances, anno):
    anno_net = {}
    zero_tensor = torch.tensor([0])

    if 'keypoints' in anno.keys():
        try:
            anno_net['keypoints'] = list(
                instances.pred_keypoints.numpy()[0].reshape(1, -1)[0])
        except IndexError:
            anno_net['keypoints'] = zero_tensor
    if 'segmentation' in anno.keys():
        try:
            anno_net['segmentation'] = [
                list(Mask(instances.pred_masks.numpy()[0]).polygons()[0])]
        except IndexError:
            anno_net['segmentation'] = [zero_tensor]
    if 'bbox' in anno.keys():
        try:
            tlbr = list(instances.pred_boxes.tensor.numpy()[0])
            top, left, bottom, right = tlbr
            width, height = right - left, bottom - top
            anno_net['bbox'] = [top, left, height, width]
        except IndexError:
            anno_net['bbox'] = zero_tensor
    return anno_net


def get_main_cfg(path, cat):
    # get standard configurations
    cfg = get_cfg()
    # select the right network
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

    # select datasets
    cfg.DATASETS.TRAIN = (f"my_dataset_train_{cat}",)
    cfg.DATASETS.TEST = (f"my_dataset_valid_{cat}",)
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.MASK_ON = False
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.2
    cfg.OUTPUT_DIR = f"{path}/output"

    return cfg


def get_advanced_cfg(path, cat, mask_it, keypoint_names):
    """add some predefined parameters"""
    # get standard configurations
    cfg = get_cfg()
    # select the right network
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_BACKBONE))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_BACKBONE)

    # select datasets
    cfg.DATASETS.TRAIN = (f"my_dataset_train_{cat}",)
    cfg.DATASETS.TEST = (f"my_dataset_valid_{cat}",)
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 30000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.MASK_ON = mask_it
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.2
    cfg.OUTPUT_DIR = f"{path}/output"

    # keypoint specific
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(keypoint_names)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [1 for _ in keypoint_names]
    return cfg


def train_main_detector(train, cat='main'):
    """train the main detector to distribute the images from"""
    path = f'../jsons/{cat}'
    pic_path = '../images'

    # Register the datasets
    modes = ['train', 'valid', 'test']
    create_local_directory(path)

    for mode in modes:
        json_file_removed(path, mode)
        register_coco_instances(
            f"my_dataset_{mode}_{cat}", {}, os.path.join(
                path, f"{mode}.json"), pic_path
        )
    cfg = get_main_cfg(path, cat)

    if train:
        # Select Trainer
        shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    # load the resulting model
    model_str = MODEL_NAME
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)

    # now perform deletions:
    shutil.rmtree(pic_path, ignore_errors=True)
    for loc_file in os.listdir(cfg.OUTPUT_DIR):
        if model_str.endswith('.png'):
            continue
        if model_str not in loc_file:
            os.remove(f'{cfg.OUTPUT_DIR}/{loc_file}')


def json_file_removed(path, mode):
    # load the original json file with all keys
    jsonfile = f"../{mode}.json"
    with open(jsonfile)as jf:
        data = json.load(jf)

    rem_keys_cat = ["file_load", "keypoints",
                    "keypoints_flip_map", "file_load"]
    for cat in data['categories']:
        [cat.pop(key, None) for key in rem_keys_cat]

    rem_keys_anno = ["num_keypoints", "keypoints", "segmentation"]
    for anno in data['annotations']:
        [anno.pop(key, None) for key in rem_keys_anno]

    with open(f'{path}/{mode}.json', 'w') as fp:
        json.dump(data, fp, indent=2)


def train_all_subnetworks(train, catnames):
    """main function for training"""

    for cat in catnames:
        # Register datasets
        path = f'../jsons/{cat}'
        pic_path = f'{path}/images'

        # Register the datasets
        modes = ['train', 'valid', 'test']
        for mode in modes:
            jsonfile = os.path.join(path, f"{mode}.json")
            with open(jsonfile)as jf:
                data = json.load(jf)

            keypoint_names = data['categories'][0]['keypoints']
            keypoint_flip_map = data['categories'][0]['keypoints_flip_map']

            register_coco_instances(
                f"my_dataset_{mode}_{cat}", {}, os.path.join(
                    path, f"{mode}.json"), pic_path
            )
            MetadataCatalog.get(
                f"my_dataset_{mode}_{cat}").keypoint_names = keypoint_names
            MetadataCatalog.get(
                f"my_dataset_{mode}_{cat}").keypoint_flip_map = keypoint_flip_map

        mask_it = True if "segmentation" in data['annotations'][0].keys(
        ) else False
        cfg = get_advanced_cfg(path, cat, mask_it, keypoint_names)

        if train:
            # Select Trainer
            shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()

        # load the resulting model
        model_str = MODEL_NAME
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)
        predictor = DefaultPredictor(cfg)

        # draw some result images
        for i in range(5):
            img = update(path, predictor, i)
            img.save(f'{cfg.OUTPUT_DIR}/{i}.png')

        # now perform deletions:
        delete_dir_models(pic_path)


def delete_dir_models(pic_path):
    """delete the local model files, to save space on disk"""
    shutil.rmtree(pic_path, ignore_errors=True)


# %%
model_list = [
    'model_0004999.pth',
    'model_0009999.pth',
    'model_0014999.pth',
    'model_0019999.pth',
    'model_0024999.pth',
    'model_final.pth',
]
eval_model_cat_dict = {
    'H': model_list[2],
    'F': model_list[1],
    'T_low': model_list[5],
    'T_up': model_list[0],
    'Fi_low': model_list[3],
    'Fi_up': model_list[5],
    'S': model_list[1],
    'A_up': model_list[3],
    'A_low': model_list[4],
    'O': model_list[1],
    'K': model_list[1],
    'F_t': model_list[4],
}


class Evaluator(object):
    """Class object to load and annotate image files for analysis"""

    def __init__(self, gen_path='../', mode='test', mask=True, truelabel=True,
                 onlyimage=False, angle_vis=False, save=False,
                 from_eval_model=False, ocr_analysis=True, draw_annotations=True, ext=False,
                 use_overall_image=False, extra_analysis=True):
        """assign constant parameters"""
        self.gen_path = gen_path  # parent path
        self.jsons_path = f'{gen_path}/jsons'  # path to the expert networks
        self.jsons = CATNAMES[::-1]  # list of categories
        self.mode = mode  # train, valid, or test
        self.mask = mask  # draw the mask as well
        self.truelabel = truelabel  # is truelabel provided?
        self.onlyimage = onlyimage  # return the image only
        self.angle_vis = angle_vis  # draw all angles on image
        self.save = save  # save the local image
        self.model_name = MODEL_NAME  # path of the model to use
        self.from_eval_model = from_eval_model  # use the best fit model from eval
        self.ocr_analysis = ocr_analysis  # detect ruler digits
        self.draw_annotations = draw_annotations  # draw annotations on image
        self.ext = ext  # use the external mode
        self.use_overall_image = use_overall_image  # use the overall image
        self.extra_analysis = extra_analysis  # use the extra analysis

        if mode in ['train', 'valid', 'test']:
            self.get_overall_data(mode)

    def __call__(self, idx, optimize_area=False, use_net=True, mode=2):
        try:
            return self.forward(idx, optimize_area, use_net, mode)
        except IndexError:
            print("optimize area failed")
            return self.forward(idx, False, use_net, mode)

    def forward(self, idx, optimize_area=False, use_net=True, mode=2):
        """perform the analysis"""
        # load the infos to the image
        self.get_img_infos(idx)
        # clear all annotations and make empty
        self.prepare_empyt_annotation()
        # detect all parts on image
        if optimize_area:
            self.detect_all_relevant_image_parts()
        # detailed analysis on the image
        self.analyse_overall_image(idx, mode=mode)
        # eventually analyse rulers:
        self.add_ocr_analysis()
        # decide what to do with results
        self.handle_results(use_net=use_net)
        # return relevant results
        return self.return_results(idx)

    def forward_raw(self, img_path, optimize_area=True, mode=2):
        self.get_raw_image(img_path)
        self.prepare_empyt_annotation()

        if optimize_area:
            self.detect_all_relevant_image_parts()

        self.analyse_overall_image(0, mode=mode)
        self.add_ocr_analysis()
        self.handle_results(use_net=True)
        return self.return_results(img_path)

    def get_raw_image(self, img_path):
        """analyse the image without any preprocessing"""
        # load the infos to the image
        self.get_img_infos(0)

        self.img_name = img_path

        # get the picture
        self.overall_img = cv2.imread(self.img_name)
        self.overall_height = self.overall_img.shape[0]
        self.overall_width = self.overall_img.shape[1]

        self.img = Image.fromarray(self.overall_img, 'RGB')
        self.draw = ImageDraw.Draw(self.img, 'RGBA')

    def get_overall_data(self, mode):
        """load the annotation data of test"""
        overall_json = f'{self.gen_path}{mode}.json'
        # 1.
        with open(overall_json) as jf:
            self.overall_data = json.load(jf)

    def get_img_infos(self, idx):
        """load the image infos"""
        img_infos = self.overall_data['images']
        img_id_map = {}
        for i, img_info in enumerate(img_infos):
            img_id_map[img_info["id"]] = i

        self.img_name = img_infos[idx]['file_name']
        self.img_name = self.img_name.replace('../', self.gen_path)

        # get the picture
        self.overall_img = cv2.imread(self.img_name)
        self.overall_height = self.overall_img.shape[0]
        self.overall_width = self.overall_img.shape[1]

        self.img = Image.fromarray(self.overall_img, 'RGB')
        self.draw = ImageDraw.Draw(self.img, 'RGBA')

    def prepare_empyt_annotation(self):
        """init the overall annotations"""
        if self.onlyimage:
            self.jsons = tqdm(CATNAMES[::-1])

        self.all_annos = {}
        self.all_annos_net = {}

        # further track left and right annotations
        self.l_r_detect = False
        self.l_r_fail = False
        self.all_annos_net_l = {}
        self.all_annos_net_r = {}
        self.res_boxes = {}

    def analyse_overall_image(self, idx, mode=1):
        """code for analysis of the overall image"""
        for cat in self.jsons:
            self.prep_local_image(idx, cat, mode=mode)
            self.analyse_local_image(cat)

    def detect_all_relevant_image_parts(self):
        """detect the relevant part of the image by using the general detector"""
        main_path = f'{self.jsons_path}/main'
        cfg = get_main_cfg(main_path, 'main')
        model_str = MODEL_NAME
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)
        predictor = DefaultPredictor(cfg)

        # evaluate the whole image
        with torch.no_grad():
            outputs = predictor(self.overall_img)

        # create directory with all boxes for each class
        self.res_boxes = {}
        instances = outputs["instances"].to("cpu")
        for num in range(len(instances.pred_classes)):
            cur_class = CATNAMES[instances.pred_classes[num]]

            if cur_class not in self.res_boxes.keys():
                self.res_boxes[cur_class] = instances.pred_boxes[num][0].tensor[0]

    def prep_local_image(self, idx, cat, mode=1):
        """load the local image and annotations"""
        if self.onlyimage:
            self.jsons.set_postfix_str(
                f'Appling the Neural Network for {cat} detection on image {idx}.')

        self.local_path = f'{self.jsons_path}/{cat}'
        jsonpath = f'{self.local_path}/{self.mode}.json'
        with open(jsonpath) as jf:
            data = json.load(jf)

        self.mask_it = True if "segmentation" in data['annotations'][0].keys(
        ) else False

        self.cat_infos = data["categories"]

        # get the relevant range
        r_min, r_max = data['infos']['range']

        # get cur_keypoint_names
        self.keypoint_names = data['categories'][0]['keypoints']

        # get the annotation
        self.anno = data['annotations'][idx]

        # get the local range
        if mode == 1:
            self.y_min, self.y_max = self.optimize_image_range(
                cat, r_min, r_max)
        else:
            self.y_min, self.y_max = self.optimize_image_range_trainstats(
                cat, r_min, r_max)

        # get local image
        if self.use_overall_image:
            self.y_min = 0
            self.y_max = self.overall_height

        self.cur_img = self.overall_img[self.y_min:self.y_max, :, :]

    def optimize_image_range(self, cat, r_min, r_max):
        """optimize the y-range in which the object is found"""

        # get the local range
        y_min = int(r_min * self.overall_height)
        y_max = int(r_max * self.overall_height)

        self.y_min_org = y_min
        self.y_max_org = y_max

        # check if object was found at all
        # or the optimization is turned on!
        if cat in self.res_boxes.keys():
            det_bbox = self.res_boxes[cat]
            y_min_det = det_bbox[1]
            y_max_det = det_bbox[3]

            disty1 = int(y_min_det - y_min)
            disty2 = int(y_max - y_max_det)

            # check if object is within expected range
            if disty1 > 0 and disty2 > 0:
                distmin = max(min(disty1, disty2), 60)
                y_min_new = max(y_min_det - distmin, 0)
                y_max_new = min(y_max_det + distmin, self.overall_height)

                y_min = y_min_new
                y_max = y_max_new

        return int(y_min), int(y_max)

    def optimize_image_range_trainstats(self, cat, r_min, r_max):
        """optimize the image range using the trainstats"""
        # get the local range
        y_min = int(r_min * self.overall_height)
        y_max = int(r_max * self.overall_height)

        self.y_min_org = y_min
        self.y_max_org = y_max

        # check if object was found at all
        # or the optimization is turned on!
        if cat in self.res_boxes.keys():
            det_bbox = self.res_boxes[cat]
            x1, y1, x2, y2 = det_bbox
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)

            det_center = [center_x, center_y]
            _, y_range = get_img_cut_dims(
                det_center, cat, [y_min, y_max], gen_path=self.gen_path)

            y_min_new = max(y_range[0], 0)
            y_max_new = min(y_range[1], self.overall_height)

            y_min = y_min_new
            y_max = y_max_new

        return int(y_min), int(y_max)

    def split_cur_image_left_right(self, cat, rang=0.45):
        """
        idea is to split the image in a left and right section before analysis
         -> by setting parts of the image to black (=0)
        """
        self.x_min = int(rang * self.overall_width)

        self.left_img = self.cur_img.copy()
        if cat != 'K':  # the sphere will only appear once!
            self.left_img[:, 0:self.x_min, :] = 0

        self.right_img = self.cur_img.copy()
        if cat != 'K':  # the sphere will only appear once!
            self.right_img[:, self.overall_width - self.x_min:, :] = 0

    def get_single_instance(self, outputs, x_min):
        instances1 = outputs["instances"].to("cpu")[:1]
        anno_net = instance2anno(instances1, self.anno)
        anno_net = transform_instances(anno_net, self.y_min, x_min=x_min)
        return anno_net

    def analyse_local_image(self, cat):
        """perform the actual analysis"""

        # load local network
        cfg = get_advanced_cfg(self.local_path, cat,
                               self.mask_it, self.keypoint_names)

        # select local model for analysis
        if self.from_eval_model:
            model_str = eval_model_cat_dict[cat]
        else:
            model_str = self.model_name

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)
        predictor = DefaultPredictor(cfg)

        # evaluate image
        with torch.no_grad():
            outputs = predictor(self.cur_img)

        l_r_previous = self.l_r_detect

        self.l_r_detect, self.l_r_fail, self.anno_net_l, self.anno_net_r = get_multiple_instances(
            self.l_r_detect, self.l_r_fail, outputs, self.anno, self.y_min)

        if cat == 'S' and l_r_previous == False:
            self.l_r_detect = False

        # check if left/right mode is active:
        if self.l_r_detect:
            self.split_cur_image_left_right(cat)
            with torch.no_grad():
                outputs_l = predictor(self.left_img)
                outputs_r = predictor(self.right_img)
            self.anno_net_l = self.get_single_instance(outputs_l, 0)
            self.anno_net_r = self.get_single_instance(outputs_r, 0)

        instances = outputs["instances"].to("cpu")[:1]
        anno_net = instance2anno(instances, self.anno)

        # shift by y
        self.anno = transform_instances(self.anno, self.y_min_org)
        anno_net = transform_instances(anno_net, self.y_min)

        # f1 analysis
        if self.extra_analysis:
            anno_net = self.extra_analysis_f12(cat, anno_net)

        # always fill to be sure
        self.all_annos_net_l[cat] = anno_net
        self.all_annos_net_r[cat] = anno_net

        self.all_annos[cat] = self.anno
        self.all_annos_net[cat] = anno_net

        # draw the true label
        if self.truelabel:
            if self.draw_annotations:
                draw_labels(self.cat_infos, self.anno, self.draw, self.mask)

        # draw the labels of the network
        if self.l_r_detect == True:
            self.all_annos_net_l[cat] = self.anno_net_l
            self.all_annos_net_r[cat] = self.anno_net_r

            # special add for ft
            self.add_ft_to_dual_analysis(cat)

            if self.draw_annotations:
                draw_network_labels(
                    self.cat_infos, self.anno_net_l, self.draw, self.mask)
                draw_network_labels(
                    self.cat_infos, self.anno_net_r, self.draw, self.mask)
        else:
            if self.draw_annotations:
                draw_network_labels(self.cat_infos, anno_net, self.draw, self.mask)

    def extra_analysis_f12(self, cat, anno_net):
        """add a local analysis for the f1 point"""

        if cat == 'F':
            rang_up = 40
            rang_down = 90

            key_nums = [1, 2]

            for key_num in key_nums:
                rang_up = int(rang_up / key_num)
                rang_down = int(rang_down / key_num)

                if self.l_r_detect:
                    t_anno_l = self.all_annos_net_l['T_low']
                    t_anno_r = self.all_annos_net_r['T_low']

                    self.anno_net_r = self.perform_kfilt_analysis(
                        self.anno_net_r, key_num, rang_up, rang_down, comp_anno=t_anno_r)
                    self.anno_net_l = self.perform_kfilt_analysis(
                        self.anno_net_l, key_num, rang_up, rang_down, comp_anno=t_anno_l)

                t_anno = self.all_annos_net['T_low']
                anno_net = self.perform_kfilt_analysis(
                    anno_net, key_num, rang_up, rang_down, comp_anno=t_anno)

        if cat == 'K':
            rang_up = 20
            rang_down = 20

            key_nums = [2]
            loc_args = ['min']

            for key_num, loc_arg in zip(key_nums, loc_args):
                rang_up = int(rang_up / key_num)
                rang_down = int(rang_down / key_num)

                if self.l_r_detect:

                    self.anno_net_r = self.perform_kfilt_analysis(
                        self.anno_net_r, key_num, rang_up, rang_down, loc_arg=loc_arg)
                    self.anno_net_l = self.perform_kfilt_analysis(
                        self.anno_net_l, key_num, rang_up, rang_down, loc_arg=loc_arg)

                anno_net = self.perform_kfilt_analysis(
                    anno_net, key_num, rang_up, rang_down, loc_arg=loc_arg)

        return anno_net

    def perform_kfilt_analysis(self, anno, key_num=1, rang_up=50, rang_down=50, lr_add=10, verbose=False, comp_anno=None, loc_arg='min'):
        img = self.overall_img.copy()
        f1 = anno['keypoints'][(key_num-1)*3:(key_num-1)*3+2]

        if len(f1) < 2:
            return anno

        f1 = [int(f1[0]), int(f1[1])]
        img_arr = img[list(
            range(f1[1]-rang_up, f1[1]+rang_down)), f1[0]-lr_add:f1[0]+lr_add, :]
        img_avg = np.mean(img_arr, axis=1)
        img_avg = np.mean(img_avg, axis=1)
        img_filt = t1filt(img_avg)
        kfiltfilt = kantenfilt(img_filt)

        corr = list(range(len(kfiltfilt)))
        for i in range(len(kfiltfilt)):
            if i < rang_up:
                corr[i] = 0
            else:
                corr[i] = (1.5 / rang_down) * (i - rang_up)

            kfiltfilt[i] += corr[i]

        if loc_arg == 'min':
            diff_y = np.argmin(kfiltfilt) - rang_up
        else:
            diff_y = np.argmax(kfiltfilt) - rang_up

        if verbose:
            plt.figure(figsize=(8, 2))
            x_arr = range(0, len(kfiltfilt))
            plt.plot(x_arr, kfiltfilt)
            plt.ylim([np.min(kfiltfilt) - 0.5, np.max(kfiltfilt) + 0.5])
            plt.vlines(rang_up, -500, 500, linestyles="dotted", colors="b")
            plt.vlines(diff_y + rang_up, -500, 500,
                       linestyles="dotted", colors="r")
            plt.xlabel('pixel in y', fontsize=14)
            plt.ylabel('Edge Filter', fontsize=14)
            plt.grid()
            plt.show()

            loc_img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(loc_img)

            draw.line((0, rang_up, rang_down, rang_up),
                      fill=(0, 100, 145, 100))
            draw.line((0, diff_y + rang_up, rang_down,
                       diff_y + rang_up), fill=(255, 0, 0, 100))

            img_arr = np.array(loc_img)
            plt.imshow(img_arr[:, :, :])

        # analyse the points 1-2 from comp anno
        if comp_anno is not None:
            t1 = comp_anno['keypoints'][(key_num-1)*3:(key_num-1)*3+2]
            if t1[1] < f1[1] + diff_y + 15:
                print(f'stopped optimization of f{key_num}')
                diff_y = 0

        anno['keypoints'][(key_num-1)*3+1] += diff_y
        return anno

    def add_ft_to_dual_analysis(self, cat):
        if cat == 'F_t' and self.l_r_detect:
            self.all_annos_net_l['F_t2'] = self.anno_net_r
            self.all_annos_net_r['F_t2'] = self.anno_net_l

    def handle_results(self, use_net=True):
        """deal with overall results and decide whether to use l/r and angle_calculations"""
        # only if we want to see all angles
        if self.angle_vis:
            if self.l_r_detect == True:
                leftright, success = get_left_right_excel(
                    self.img_name, self.ext)
                if success:
                    self.all_annos_net = self.all_annos_net_l if leftright else self.all_annos_net_r
                else:
                    jcla_l = draw_all_angles(
                        self.overall_img, self.draw, self.all_annos_net_l, get_jcla=True, acc_boost=use_net)
                    jcla_r = draw_all_angles(
                        self.overall_img, self.draw, self.all_annos_net_r, get_jcla=True, acc_boost=use_net)

                    self.all_annos_net = self.all_annos_net_l if jcla_l > jcla_r else self.all_annos_net_r

            self.draw_angles(use_net)

    def draw_angles(self, use_net=True):
        if use_net:
            draw_all_angles(self.overall_img, self.draw,
                            self.all_annos_net, acc_boost=True)
        else:
            draw_all_angles(self.overall_img, self.draw,
                            self.all_annos, acc_boost=False)

    def return_results(self, idx):
        """finally return all relevant results"""
        if self.save:
            self.img.save(f'../results/cur_img_{idx}.png')

        if self.onlyimage:
            return self.img

        if self.ext:
            return self.img, self.all_annos_net

        return self.img, self.all_annos, self.all_annos_net

    def analyse_single_image_single_type(self, cat, img_name, mask_it=False):
        self.prepare_empyt_annotation()

        self.overall_img = cv2.imread(img_name)
        self.overall_height = self.overall_img.shape[0]

        self.img = Image.fromarray(self.overall_img, 'RGB')
        self.draw = ImageDraw.Draw(self.img, 'RGBA')

        path = f'{self.jsons_path}/{cat}'
        jsonpath = f'{path}/{self.mode}.json'
        with open(jsonpath) as jf:
            data = json.load(jf)
        # dummy annotation
        anno = data['annotations'][0]

        self.cat_infos = data["categories"]
        self.keypoint_names = data['categories'][0]['keypoints']

        path = f'{self.jsons_path}/{cat}'
        cfg = get_advanced_cfg(path, cat, mask_it, self.keypoint_names)
        model_str = MODEL_NAME
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str)
        predictor = DefaultPredictor(cfg)

        with torch.no_grad():
            outputs = predictor(self.overall_img)

        instances = outputs["instances"].to("cpu")[:1]
        anno_net = instance2anno(instances, anno)
        self.all_annos_net[cat] = anno_net

        draw_network_labels(self.cat_infos, anno_net, self.draw, self.mask)
        return self.img

    def add_ocr_analysis(self):
        if self.ocr_analysis:
            imgfac = ocr_analysis(self.overall_img)
            self.all_annos_net['imgfac'] = imgfac
            self.all_annos_net_l['imgfac'] = imgfac
            self.all_annos_net_r['imgfac'] = imgfac
            self.all_annos['imgfac'] = imgfac


def get_num_from_name(name: str):
    """name -> num"""
    return int(name.split(
        '/')[2].split('.')[0].split('_')[0].replace('S', '').replace('n', ''))


def dfnum(idx: int):
    """num -> df_num"""
    return idx - 1


def get_left_right_excel(img_name: str, ext=False):
    """look on the excel and decide the side to use for evaluation"""
    try:
        if not ext:
            num = get_num_from_name(img_name)
            df = pd.read_excel('../osteo.xlsx')
            df_res = df.iloc[dfnum(num)]
            seite = df_res['Seite']

        else:
            data = pd.read_excel('../extern_data.xlsx')
            leftrightlist = list(data['Pat. Nummer'])
            img_num = int(img_name.split('/')[-1].split('.')[0])
            num = leftrightlist.index(img_num)
            seite = data['Seite '][num]

        if seite in ['LI', 'L', 'Links', 'LEFT', 'LINKS']:
            return True, True
        if seite in ['RE', 'R', 'Rechts', 'RIGHT', 'RECHTS']:
            return False, True
        return False, False

    except:
        print('l/R detection failed')
        return False, False


def sufficient_distance(left1, left2, krit_distance=150):
    """check if the difference between instances is large enough"""
    if abs(left1 - left2) < krit_distance:
        return False
    return True


def get_multiple_instances(l_r_detect, l_r_fail, outputs, anno, y_min, threshold=0.98):
    """
    take the output of the neural network and take multiple instances by score
    Idea:
    1. Detect if multiple objects can be detected above a threshold.
    2. If that is true -> set l_r_detect to true
    3. Extract both objects and group them by left (large x) and right (small x)
    """
    instances1 = outputs["instances"].to("cpu")[:1]
    anno_net_1 = instance2anno(instances1, anno)
    anno_net_1 = transform_instances(anno_net_1, y_min)
    left1 = anno_net_1['bbox'][0]

    instances2 = outputs["instances"].to("cpu")[1:2]

    if instances2.scores.nelement() >= 1:
        anno_net_2 = instance2anno(instances2, anno)
        anno_net_2 = transform_instances(anno_net_2, y_min)
        left2 = anno_net_2['bbox'][0]

        if (instances2.scores[0] > threshold) and sufficient_distance(left1, left2):
            l_r_detect = True

    else:
        l_r_fail = True

    if l_r_fail == False and l_r_detect == True:

        # x of first annotation large -> anno1 is left and anno2 is right
        if left1 > left2:
            anno_net_l = anno_net_1
            anno_net_r = anno_net_2

        # x of second annotation large -> anno2 is left and anno1 is right
        if left1 < left2:
            anno_net_l = anno_net_2
            anno_net_r = anno_net_1
    else:
        anno_net_l = anno_net_1
        anno_net_r = anno_net_1

    return l_r_detect, l_r_fail, anno_net_l, anno_net_r


def transform_instances(anno, y_min, x_min=0):
    """add the offset to the annotations"""
    new_anno = transform_bbox(anno, y_min, x_min)
    new_anno = transform_segmentation(new_anno, y_min, x_min)
    new_anno = transform_keypoints(new_anno, y_min, x_min)
    return new_anno


def transform_bbox(anno, y_min, x_min, key='bbox'):
    new_anno = anno.copy()
    if key in anno.keys() and (len(anno[key]) > 2):
        x, y, width, height = anno[key]
        new_anno[key] = [x + x_min, y + y_min, width, height]

    return new_anno


def transform_segmentation(anno, y_min, x_min, key='segmentation'):
    new_anno = anno.copy()
    if key in anno.keys():
        segmentation = anno[key]
        new_segmentation = []
        for poly in segmentation:
            if len(poly) > 2:
                x_arr, y_arr = poly[::2], poly[1::2]
                x_arr = [x + x_min for x in x_arr]
                y_arr = [y + y_min for y in y_arr]
                new_seg = []
                for x, y in zip(x_arr, y_arr):
                    new_seg.append(x)
                    new_seg.append(y)
                new_segmentation.append(new_seg)
        new_anno[key] = new_segmentation

    return new_anno


def transform_keypoints(anno, y_min, x_min, key='keypoints'):
    new_anno = anno.copy()
    if key in anno.keys():
        keypoint_arr = anno[key]
        if len(keypoint_arr) > 2:
            x_arr = keypoint_arr[0::3]
            y_arr = keypoint_arr[1::3]
            x_arr = [x + x_min for x in x_arr]
            y_arr = [y + y_min for y in y_arr]
            new_anno[key][0::3] = x_arr
            new_anno[key][1::3] = y_arr

    return new_anno


# %%
if __name__ == '__main__':
    evaluator = Evaluator(onlyimage=True, angle_vis=True,
                          use_overall_image=True)
    img = evaluator(15)
# %%
