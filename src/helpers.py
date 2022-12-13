# %%
import json
from datetime import date
from random import shuffle
import os
import cv2
from matplotlib import pyplot as plt
import nrrd
from PIL import Image
import numpy as np
import pandas as pd
from imantics import Mask
from categories import SEG_NAMES, CATEGORIES
from tqdm import tqdm
import math

# preparation constants
ANN_KEY = 'annotations'
IMG_KEY = 'images'

# excel constants
EVAL_KEYS = ['MAD', 'Mikulicz auf TP', 'mFTA\n(°)', 'mLDFA\n(°)', 'mMPTA\n(°)',
             'JLCA\n(°)', 'mLPFA', 'AMA', 'mLDTA']
EXCEL_PATH = '../osteo.xlsx'

# dataset split
SPLIT = {
    'test': 0.3,
    'valid': 0.1,
    'train': 0.6,
}


def get_patient_files(patient: str) -> dict:
    """get all relevant files for one patient"""
    files = os.listdir(patient)

    imgname = list(filter(lambda x: x[:2] == 'EE', files))[0]
    segname = list(filter(lambda x: x.replace(
        '_23', '').endswith('label.nrrd'), files))[0]
    segname_full = list(filter(lambda x: x.replace(
        '_23', '').endswith('seg.nrrd'), files))[0]

    patfiles = {
        'img': f'{patient}/{imgname}',
        'segname': f'{patient}/{segname}',
        'segname_full': f'{patient}/{segname_full}'
    }
    for seg in SEG_NAMES:
        seglist = list(filter(lambda x: x.endswith('.mrk.json'), files))
        segname = list(filter(lambda x: f'{seg}_' in x, seglist))[0]
        patfiles[seg] = f'{patient}/{segname}'

    return patfiles


def get_image(img_path: str, maxval=0, rescale=255):
    """get an image in PIL image format"""
    img_arr = nrrd.read(img_path)[0][:, :, 0]

    # set maxval if desired
    if maxval == 0:
        img_arr = np.transpose(img_arr)
    else:
        img_arr = np.transpose(
            np.uint8(np.round((img_arr / maxval) * rescale)))

    img = Image.fromarray(img_arr)
    return img, img_arr


def merge_imgs(background: Image, foreground: Image):
    """produce a merged image from foreground and background"""
    merged = background.copy()
    merged.paste(foreground, (0, 0), foreground)
    return merged


def test_merge(img_path: str, seg_path: str, maxval=4096):
    """test if the setup works correct"""
    img, _ = get_image(img_path, maxval)
    seg, _ = get_image(seg_path, 7)
    seg = seg.convert("RGBA")
    merged = merge_imgs(img, seg)
    return merged


def get_fac(patfiles):
    """get the factor for merging keypoints"""
    _, seg_arr = get_image(patfiles['segname'], 0)
    x_h1mark, y_h1mark = read_key_coo(patfiles, key='H')
    x_h1, y_h1 = read_h_seg_coo(seg_arr)

    # calculate scale factors
    fac_x = x_h1 / x_h1mark
    fac_y = y_h1 / y_h1mark

    return [fac_x, fac_y], seg_arr


def read_key_coo(patfiles, key):
    """get the position of the keypoints"""
    res = pd.read_json(patfiles[key])
    x_mark, y_mark = res['markups'][0]['controlPoints'][0]['position'][:2]
    return x_mark, y_mark


def read_multiple_key_coo(patfiles, keys, fac, allowed_names):
    """get all position of the keypoints"""

    keypoint_arr = []
    len_k = 0

    for key in keys:
        res = pd.read_json(patfiles[key])

        for position in res['markups'][0]['controlPoints']:
            cur_name = position["label"]
            if cur_name in allowed_names:
                len_k += 1
                x, y = position["position"][:2]
                x, y = x * fac[0], y * fac[1]
                keypoint_arr.append(x)
                keypoint_arr.append(y)
                keypoint_arr.append(2)  # visibility true

    return keypoint_arr, len_k


def read_h_seg_coo(seg_arr, val=1):
    """read h coo from segmentation"""
    h1seg = veccheck(seg_arr, val)
    y_h1, x_h1 = np.where(h1seg)
    x_h1 = float(np.mean(x_h1))
    y_h1 = float(np.mean(y_h1))
    return x_h1, y_h1


def checkval(x, val):
    if x == val:
        return True
    return False


veccheck = np.vectorize(checkval)


def bbox2(img):
    """
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    bbox = tlbr2bbox(top, left, bottom, right)
    return bbox


def tlbr2bbox(top, left, bottom, right, op=int):
    """
    tlbr = [top, left, bottom, right]
    to ->
    bbox = [x(left), y(top), width, height]
    """
    x = op(left)
    y = op(top)
    width = op(right - left)
    height = op(bottom - top)

    return [x, y, width, height]


def bbox_from_keypointarr(keypoint_arr: list):
    """fromat of arr: [x, y, 2, x1, y1, 2, ...]"""
    x_arr = keypoint_arr[::3]
    y_arr = keypoint_arr[1::3]

    left, right = min(x_arr), max(x_arr)
    top, bottom = min(y_arr), max(y_arr)

    bbox = tlbr2bbox(top, left, bottom, right)
    area = (bottom-top) * (right - left)
    return bbox, area


def get_fullseg_infos(seg_infos, search_name):
    """return dict containing all infos"""
    # we need extend, label_value, layer

    # check if search_name exists:
    all_keys = seg_infos.keys()
    rel_keys = [key for key in all_keys if key.endswith('_Name')]
    rel_key = [key for key in rel_keys if seg_infos[key] == search_name][0]
    num = int(rel_key.split('_')[0].replace('Segment', ''))

    # extract relevant infos:
    label_value = int(seg_infos[f'Segment{num}_LabelValue'])
    layer = int(seg_infos[f'Segment{num}_Layer'])
    extent = [int(val)
              for val in seg_infos[f'Segment{num}_Extent'].split(' ')[:4]]
    off_x = extent[0]
    off_y = extent[2]

    return {
        'label_value': label_value,
        'layer': layer,
        'extent': extent,
        'off_x': off_x,
        'off_y': off_y,
        'shift': [off_x, off_y]
    }


def get_segmentation(loc_infos: dict, seg_arr):
    """
    get the segmentation from the local infos:
    sing the "layer" and the "label_value" info
    """
    loc_seg = seg_arr[loc_infos['layer'], :, :, 0]
    loc_seg = veccheck(loc_seg, loc_infos['label_value'])
    loc_seg = loc_seg.transpose()
    loc_seg = loc_seg.astype(np.uint8)

    return loc_seg


def factor_segmentation(patfiles):
    """scale the segmentation using the .label file as orientation"""

    seg_arr, seg_infos = nrrd.read(patfiles['segname_full'])

    loc_infos_h = get_fullseg_infos(seg_infos, 'Hftkopf')
    loc_seg_h = get_segmentation(loc_infos_h, seg_arr)

    _, seg_arr_old = get_image(patfiles['segname'], 0)

    x_h1_old, y_h1_old = read_h_seg_coo(seg_arr_old)
    x_h1_new, y_h1_new = read_h_seg_coo(loc_seg_h)

    fac_x = y_h1_old / y_h1_new
    fac_y = x_h1_old / x_h1_new

    return [fac_x, fac_y], seg_arr, seg_infos


def get_annotation_dicts_seg(patient: str, patient_id: int):
    """get all segmentations from the .seg file"""
    patfiles = get_patient_files(patient)
    key_fac, _ = get_fac(patfiles)
    seg_fac, seg_arr, seg_infos = factor_segmentation(patfiles)
    ann_list = []

    # go trough all individual segmentations
    for i, cat in enumerate(CATEGORIES):

        cat_id = cat['id']
        idx = len(CATEGORIES) * patient_id + i
        cur_seg_name = cat["file_load"]

        # add the keypoints
        allowed_names = cat["keypoints"]

        keypoint_arr, len_k = read_multiple_key_coo(
            patfiles, cur_seg_name, key_fac, allowed_names)

        cur_bbox, area = bbox_from_keypointarr(keypoint_arr)

        # init the dictionary
        ann_dict = {
            "id": idx,
            "image_id": patient_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "num_keypoints": len_k,
            "keypoints": keypoint_arr,
            "bbox": cur_bbox,
            "area": area,
        }

        # add the segmentation
        if "full_seg" in cat.keys():
            try:
                seg_name = cat["full_seg"]
                loc_infos_seg = get_fullseg_infos(seg_infos, seg_name)
                loc_seg = get_segmentation(loc_infos_seg, seg_arr)
                sh_seg = loc_seg.shape
                new_sh = int(sh_seg[1] * seg_fac[1]
                             ), int(sh_seg[0] * seg_fac[0])
                cur_seg = cv2.resize(
                    loc_seg, (new_sh), interpolation=cv2.INTER_AREA)

                new_bbox = bbox2(cur_seg)
                max_bbox = maximize_bbox(cur_bbox, new_bbox)
                poly = Mask(cur_seg).polygons().segmentation
                area = int(np.sum(cur_seg > 0))

                ann_dict["area"] = area
                ann_dict["bbox"] = max_bbox
                ann_dict["segmentation"] = poly
            except:
                print(f"{patient} {cat['full_seg']}")
                ann_dict["segmentation"] = [[]]

        ann_list.append(ann_dict)

    return ann_list


def get_annotation_dicts_label(patient: str, patient_id: int):
    """get all segmentations from the .label file"""
    patfiles = get_patient_files(patient)
    key_fac, seg_arr = get_fac(patfiles)
    ann_list = []

    # go trough all individual segmentations
    for i, cat in enumerate(CATEGORIES):
        cat_id = cat['id']
        idx = len(CATEGORIES) * patient_id + i
        cur_seg_names = cat["file_load"]

        # add the keypoints
        allowed_names = cat["keypoints"]
        keypoint_arr, len_k = read_multiple_key_coo(
            patfiles, cur_seg_names, key_fac, allowed_names)
        cur_bbox, area = bbox_from_keypointarr(keypoint_arr)

        # init the dictionary
        ann_dict = {
            "id": idx,
            "image_id": patient_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "num_keypoints": len_k,
            "keypoints": keypoint_arr,
            "bbox": cur_bbox,
            "area": area,
        }

        # add the segmentation
        if "mask_id" in cat.keys():
            seg_val = cat["mask_id"]
            cur_seg = veccheck(seg_arr, seg_val)
            new_bbox = bbox2(cur_seg)
            max_bbox = maximize_bbox(cur_bbox, new_bbox)
            poly = Mask(cur_seg).polygons().segmentation
            area = int(np.sum(cur_seg > 0))

            ann_dict["area"] = area
            ann_dict["bbox"] = max_bbox
            ann_dict["segmentation"] = poly

        ann_list.append(ann_dict)

    return ann_list


def maximize_bbox(cur_bbox, new_bbox):
    """
    take two bounding boxes and maximize
    bbox = [x(left), y(top), width, height]
    """
    # bbox 1
    left1, top1, width1, height1 = cur_bbox
    right1, bottom1 = left1 + width1, top1 + height1
    # bbox 2
    left2, top2, width2, height2 = new_bbox
    right2, bottom2 = left2 + width2, top2 + height2
    # minmax
    left = min(left1, left2)
    right = max(right1, right2)
    top = min(top1, top2)
    bottom = max(bottom1, bottom2)
    # return maximised bbox
    return tlbr2bbox(top, left, bottom, right)


def create_image_repo(paths: list, maxval=4096, exclude=True, img_path='../images'):
    """process all seg files and build png files"""
    patients = get_patients_from_paths(paths, exclude=exclude)
    err_patients = []

    for patient in tqdm(patients):

        try:
            patfiles = get_patient_files(patient)
            imgname = patfiles["img"]
            img, _ = get_image(imgname, maxval)
            new_name = get_new_image_name(patient, img_path=img_path)
            img.save(new_name)
        except IndexError:
            print('Patient Directory Empty!')
            print(patient)
            err_patients.append(patient)

    [patients.remove(err_patient) for err_patient in err_patients]

    # save the patients to avoid re-processing
    with open("./patients.txt", "w") as f:
        f.write("\n".join(patients))

    return patients


def get_new_image_name(patient: str, img_path='../images'):
    """containes the renaiming logic"""
    patfiles = get_patient_files(patient)
    imgname = patfiles["img"]
    leftright = imgname.split('/')[-2]
    new_name = f'{img_path}/{leftright}.png'
    return new_name


def get_image_dict(patient: str, pat_num: int):
    """get the correct image dicitonary"""

    new_name = get_new_image_name(patient)
    new_name = new_name.replace('.png', '')
    new_name = f'{new_name}Sn.png'
    
    img = Image.open(new_name)
    arr_img = np.array(img)
    sh_img = arr_img.shape[:2]
    img_dict = {
        "id": pat_num,
        "file_name": new_name,
        "height": sh_img[0],
        "width": sh_img[1],
    }
    return img_dict


def perform_patient_dict(patient: str, idx: int, use_seg_file=True):
    """create a directory filled with all images"""

    if use_seg_file:
        ann_dict_list = get_annotation_dicts_seg(patient, idx)
    else:
        ann_dict_list = get_annotation_dicts_label(patient, idx)

    img_dict = get_image_dict(patient, idx)

    return ann_dict_list, img_dict


def make_empty_coco(mode: str):
    """create an empyt coco formatted dict"""
    des = f'{mode}-Radiomics detection in coco-format'
    today = date.today()
    today_str = str(today.year) + str(today.month) + str(today.day)

    coco = {
        "infos": {
            "description": des,
            "version": "0.01",
            "year": today.year,
            "contributor": "Nikolas Wilhelm",
            "date_created": today_str
        },
        "licences": [
            {
                "id": 1,
                "name": "MIT"
            },
        ],
        "categories": CATEGORIES,
        IMG_KEY: [],
        ANN_KEY: [],
    }
    return coco


def get_patients_from_paths(paths: list, exclude=True):
    """create a list of patients from multiple paths"""
    patients = []

    for path in paths:
        patlist = os.listdir(path)

        loc_patlist = []
        for patient in patlist:
            if patient[0] != '.':
                loc_patlist.append(f'{path}/{patient}')

        patients.extend(loc_patlist)

    if exclude:
        err_patient = f'/media/biomech/Extreme SSD/data/Osteosyn/Segmentiert/Rechts/390S'
        patients.remove(err_patient)

    return patients


def dis_patients(patients: list, split: dict):
    """get the patients distribution"""
    shuffle(patients)
    patlen = len(patients)

    idx_count = 0
    dis = {}
    for splitname in split.keys():
        dis[splitname] = {}

        cur_len = int(patlen * split[splitname]) + idx_count

        dis[splitname]['patient'] = patients[idx_count:cur_len]
        dis[splitname]['idx'] = list(range(idx_count, cur_len))

        idx_count = cur_len

    return dis, patients


def dis_patients_annotated_test(patients: list, split: dict):
    """get the patient distribution using the pot test patients and splitting the rest"""
    # get the potential available test patients from the excel
    pot_test_pats = get_potential_test_pats()

    # get all patient numbers
    patient_nums = [int(pat.split('/')[-1].replace('S', '').replace('_K', '').replace('n', ''))
                    for pat in patients]

    # take the match between numbers and test patients
    test_pat_nums = list(set(patient_nums).intersection(pot_test_pats))
    test_len = len(test_pat_nums)
    print(test_len)

    # match the actual patients
    test_pats = []
    for pat in patients:
        if int(pat.split('/')[-1].replace('S', '').replace('_K', '').replace('n', '')) in test_pat_nums:
            test_pats.append(pat)

    # now remove the test pats from all patietns and add them to the start:
    [patients.remove(test_pat) for test_pat in test_pats]
    test_pats.extend(patients)
    patients = test_pats

    # lets construct the test distribution:
    dis = {}
    dis['test'] = {
        'patient': patients[0:test_len],
        'idx': list(range(0, test_len)),
    }

    # next the valid distribution is determined by the split:
    pat_len = len(patients)
    valid_len = int(split['valid'] * pat_len)
    dis['valid'] = {
        'patient': patients[test_len: test_len+valid_len],
        'idx': list(range(test_len, test_len+valid_len)),
    }

    # last the train distribution:
    dis['train'] = {
        'patient': patients[test_len+valid_len::],
        'idx': list(range(test_len+valid_len, pat_len)),
    }

    return dis, patients


def get_new_test_pats():
    df = pd.read_excel('../newtest.xlsx')
    pot_test_pats = []
    for _, df_loc in df.iterrows():
        pot_test_pats.append(df_loc['ID'])
    return pot_test_pats


def get_potential_test_pats(newmode=True):
    """take all pats from the excel that have all angles"""

    # from new excel
    if newmode:
        pot_test_pats = get_new_test_pats()
        return pot_test_pats

    # old excel only if all values are supplied
    df = pd.read_excel(EXCEL_PATH)
    pot_test_pats = []

    for _, df_loc in df.iterrows():

        if True in [isinstance(df_loc[key], str) for key in EVAL_KEYS]:
            continue

        if True in [math.isnan(df_loc[key]) for key in EVAL_KEYS]:
            continue

        pot_test_pats.append(df_loc['ID'])

    return pot_test_pats


def create_cocos(split: dict, from_excel_test=True, intramode=False):
    """create datasets in coco format"""

    with open('./patients.txt', 'r') as f:
        patients = f.readlines()
    patients = [p.replace('\n', '') for p in patients]

    if from_excel_test:
        dis, patients = dis_patients_annotated_test(patients, split)
    else:
        dis, patients = dis_patients(patients, split)

    if intramode:
        splitname = 'intra'
        dis[splitname] = {}

        cur_len = int(len(patients))
        dis[splitname]['patient'] = patients
        dis[splitname]['idx'] = list(range(0, cur_len))

    # create a json file for train / val / test
    for dataset in dis.keys():
        coco = make_empty_coco(dataset)

        for idx, patient in tqdm(zip(dis[dataset]['idx'], dis[dataset]['patient'])):

            try:
                ann_dict_list, img_dict = perform_patient_dict(patient, idx)

                # append the dictionaries to the coco bunch
                coco[IMG_KEY].append(img_dict)
                coco[ANN_KEY].extend(ann_dict_list)
            except:
                # this is tolerated as we can have numerous errors
                print(patient)

        # save the coco
        local_path = os.getcwd()
        add = "../" if local_path[-3:] == "src" else ""
        save_file = f'{add}{dataset}.json'

        print(f'Saving to: {save_file}')
        with open(save_file, 'w') as fp:
            json.dump(coco, fp, indent=2)


def read_all_jsons(json_path: str, keys: list):
    """read all data within a directory"""
    all_data = {}
    for key in keys:
        with open(f'{json_path}/{key}.json') as jsonfile:
            data = json.load(jsonfile)
        all_data[key] = data
    return all_data


def create_local_directory(path: str):
    """local directoy creation with catch upon existence"""
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"{path} directory already exists.")


def distribute_jsons_to_single_class(keynames: list, json_path='../', json_dir='../jsons'):
    """
    (1) read the overall jsons
    (2) seprate to single jsons according to the category
    """

    create_local_directory(json_dir)
    cat_names = [cat["supercategory"] for cat in CATEGORIES]

    # go trough each category
    for cat in cat_names:
        print(cat)
        all_data = read_all_jsons(json_path, keynames)

        # create directory
        local_dir = f'{json_dir}/{cat}'
        create_local_directory(local_dir)

        all_loc_data = {}
        for key in keynames:
            # local json info
            data = all_data[key]
            # now only maintain the relevant info
            new_data = create_spezialised_json(data, cat)
            # append to the all loc dat dict
            all_loc_data[key] = new_data

        # (1) check percentage range
        min_pers, max_pers = [], []
        for loc_key in all_loc_data.keys():
            loc_data = all_loc_data[loc_key]
            min_per, max_per = get_minmax_percentage(loc_data)
            min_pers.append(min_per), max_pers.append(max_per)

        # (2) apply tolerance
        r_min, r_max = apply_tolerance(min_pers, max_pers, tol=0.0)

        for key in keynames:
            # now redistribute the images
            all_data_loc = all_loc_data.copy()
            cur_data = all_data_loc[key].copy()
            redis_data = redistribute_images(
                cur_data, r_min, r_max, local_dir)
            redis_data['infos']['range'] = [r_min, r_max]
            # save new data
            with open(f'{local_dir}/{key}.json', 'w') as fp:
                json.dump(redis_data, fp, indent=2)


def apply_tolerance(min_pers, max_pers, tol=0.02):
    """
    take min and max and apply the tolreance with checking image margins
    """
    min_per = min(min_pers)
    max_per = max(max_pers)
    # add tolerance and define the crop-range
    r_min = round(max(0.0, min_per - tol), 2)
    r_max = round(min(1.0, max_per + tol), 2)
    print(f'Height Range: {r_min} - {r_max}')
    return r_min, r_max


def create_spezialised_json(old_data: dict, cat: str):
    """
    go trough json and only maintain the relevant categories and annotations
    """
    # general info copy
    new_data = {}
    new_data['infos'] = old_data['infos'].copy()
    new_data['licences'] = old_data['licences'].copy()

    # copy only relevant category
    id_list = []
    new_data['categories'] = []
    for local_cat in old_data['categories']:
        if local_cat['supercategory'] == cat:
            id_list.append(local_cat['id'])
            local_new_cat = local_cat.copy()
            local_new_cat['id'] = 1
            new_data['categories'].append(local_new_cat)

    # copy images
    new_data[IMG_KEY] = old_data[IMG_KEY].copy()

    # copy only relevant annotation
    new_data[ANN_KEY] = []
    for local_ann in old_data[ANN_KEY]:
        if local_ann['category_id'] in id_list:
            local_new_ann = local_ann.copy()
            local_new_ann['category_id'] = 1
            new_data[ANN_KEY].append(local_new_ann)

    return new_data


def redistribute_images(new_data: dict, r_min: float, r_max: float, local_dir: str):
    """
    Idea is:
    (1) create new local image folder with the crop ranged-images
    (2) modify the annotations to apply to the new crop-range
    """
    redis_data = {}
    redis_data['infos'] = new_data['infos'].copy()
    redis_data['licences'] = new_data['licences'].copy()
    redis_data['categories'] = new_data['categories'].copy()

    # (3) create new local image folder with the crop ranged-images
    new_img_dir = f'{local_dir}/images'
    create_local_directory(new_img_dir)
    image_infos = new_data['images']
    y_offsets = []
    new_image_infos = []

    for image_info in tqdm(image_infos):
        # get local infos
        loc_height = image_info['height']
        y_min = int(r_min * loc_height)
        y_max = int(r_max * loc_height)
        new_height = y_max - y_min

        # save local offset
        y_offsets.append(y_min)

        # copy image
        new_img_name = image_info['file_name'].split('/')[-1]
        img_arr = np.array(Image.open(image_info["file_name"]))
        crop_img_arr = img_arr[y_min:y_max, :]
        new_img = Image.fromarray(crop_img_arr)
        new_img.save(f'{new_img_dir}/{new_img_name}')

        # overwite dict infos
        image_info['file_name'] = new_img_name
        image_info['height'] = new_height

        # append
        new_image_infos.append(image_info)

    redis_data['images'] = new_image_infos

    # (4) modify the annotations to apply to the new crop-range
    redis_data['annotations'] = modify_y_annotations(
        y_offsets, new_data[ANN_KEY])

    return redis_data


def modify_y_annotations(y_offsets: list, annotations: dict):
    """pass trough the array of annotaions and substract y offsets"""

    annos = []
    for offset, loc_ann in zip(y_offsets, annotations):

        # modify keypoints
        new_ann = loc_ann.copy()
        new_ann['keypoints'] = loc_ann['keypoints'].copy()
        new_ann['keypoints'][1::3] = [
            y - offset for y in loc_ann['keypoints'][1::3]]
        # modify bbox
        new_ann['bbox'] = loc_ann['bbox'].copy()
        new_ann['bbox'][1] = loc_ann['bbox'][1] - offset
        # modify segmentation
        if 'segmentation' in loc_ann.keys():
            loc_seg = loc_ann['segmentation'].copy()
            for i, loc_arr in enumerate(loc_seg):
                loc_arr[1::2] = [y - offset for y in loc_arr[1::2]]
                # reassign
                loc_seg[i] = loc_arr
            new_ann['segmentation'] = loc_seg

        # append annotations
        annos.append(new_ann)

    return annos


def get_minmax_percentage(data: dict, file_info=False):
    """
    go trough all annotations and extract min and max from the keypoints and bounding box
    """
    img_infos = data[IMG_KEY]
    # create image id mapping
    img_id_map = {}
    for i, img_info in enumerate(img_infos):
        img_id_map[img_info["id"]] = i
    # (1) check percentage range
    r_mins, r_maxs = [], []
    file_max = []
    for ann in data[ANN_KEY]:
        image_id = ann['image_id']
        img_info = img_infos[img_id_map[image_id]]

        # get height, min and max pixel height
        height = img_info['height']
        min_h, max_h = get_min_max_heigth_from_keypoints(ann['keypoints'])
        y, height_bb = ann['bbox'][1], ann['bbox'][3]
        min_bb, max_bb = y, y + height_bb

        min_h = min(min_h, min_bb)
        max_h = max(max_h, max_bb)

        r_min_loc = max(min_h / height, 0)
        r_max_loc = min(max_h / height, 1)

        r_mins.append(r_min_loc)
        r_maxs.append(r_max_loc)

        file_max.append(img_info["file_name"])

    mean_max = np.mean(r_maxs)
    std_max = np.std(r_maxs)
    tol = 0.02
    std_fac = 2.5

    min_per = min(r_mins) - tol
    max_per = min(
        [1, max(mean_max + std_fac * std_max + tol, max(r_mins) + tol)])

    plt.figure(figsize=(14, 14))
    plt.plot(r_mins, color='b')
    plt.plot(min_per * np.ones(len(r_mins)), color='b', linestyle='--')
    plt.plot(r_maxs, color='r')
    plt.plot(max_per * np.ones(len(r_mins)), color='r', linestyle='--')
    plt.ylim([0, 1.05])
    plt.grid()
    plt.show()

    print(file_max[np.argmax(r_maxs)]) if file_info else None

    return min_per, max_per


def get_min_max_heigth_from_keypoints(keypoint_arr: list):
    """take every 3rd element (height) and take min and max"""
    height_vals = keypoint_arr[1::3]
    min_h, max_h = min(height_vals), max(height_vals)
    return min_h, max_h

    # %%
if __name__ == '__main__':
    # add your  personal path od the data here
    paths = ['E:/Segmentiert/Links',
             'E:/Segmentiert/Links',
             ]
    split = {
        'train': 0.6,
        'valid': 0.2,
        'test': 0.2,
    }
    patients = create_image_repo(paths, exclude=False)
    create_cocos(patients, split)
    distribute_jsons_to_single_class(split.keys())


# %%