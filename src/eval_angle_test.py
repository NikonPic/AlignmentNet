# %%
# IDEA:
# 1. evaluate picture and get anno and anno_net
# 2. get the excel data
# 3. compare: anno_net <-> (anno + exce angles) / 2
# 4. buil averages over test dataset
# 5. visualize all angles in picture
import math
import cv2
import numpy as np
import pandas as pd
import json
import helpers as hp
from train_detectron import Evaluator
from angle_calc import anno_calculation, json_prep_anno
from tqdm import tqdm
from categories import CATNAMES, CATEGORIES
import matplotlib.pyplot as plt
import pingouin as pg
from PIL import Image, ImageDraw


# constants
USE_EXCEL = True
USE_CLINIC_EXCEL = True

MODE = 'test'
DF_KEYS = ['Age at surgery', 'Größe', 'Gewicht']
DF_TEXT = ['Age at surgery', 'Height', 'Weight']
DF_METRICS = ['years', 'cm', 'kg']

# next we need the angle calc:
ANGLE_NAMES = ['Mikulicz', 'Mikulicz auf TP', 'MAD', 'mLPFA', 'AMA', 'mLDFA', 'JLCA',
               'mMPTA', 'mFTA','KJLO', 'mLDTA']

UNITS = ['mm', '%', 'mm', '°', '°', '°', '°', '°', '°', '°', '°']

PERFORM_OSTEOTOMY = False
CUT_LEN = 'Korrektur (mm)'
CUT_ANGLE = 'Winkel'

ANGLE_NAMES_POSTOP = ['Mikulicz', 'Mikulicz auf TP', 'MAD', 'mFTA', 'KJLO']

if PERFORM_OSTEOTOMY:
    ANGLE_NAMES.append(CUT_LEN)
    ANGLE_NAMES.append(CUT_ANGLE)
    UNITS.append('mm')
    UNITS.append('°')

if USE_EXCEL:
    RATERS = ['OS1', 'OS2', 'AI']
else:
    RATERS = ['OS1', 'AI']

if USE_CLINIC_EXCEL:
    RATERS = ['OS1', 'OS2', 'OS3', 'AI']

# % functions for printing dataset


def get_num_from_name(name: str):
    """name -> num"""
    return int(name.split(
        '/')[2].split('.')[0].split('_')[0].replace('S', '').replace('n', ''))


def dfnum(idx: int):
    """num -> df_num"""
    return idx - 1


def extract_df_id(idx, mode='test'):
    """Read the excel info and fill the dict"""

    # get the id by reading the mode file
    with open(f'../{mode}.json') as fp:
        data = json.load(fp)

    test_filename = data['images'][idx]['file_name']
    test_num = get_num_from_name(test_filename)

    # now find the matching id in the excel and read the data
    df = pd.read_excel(hp.EXCEL_PATH)

    # extract the data
    res_excel = df.iloc[dfnum(test_num)][hp.EVAL_KEYS]

    return res_excel


def dataset_info(modes=['test']):
    """Dataset information"""

    df = pd.read_excel(hp.EXCEL_PATH)

    df_nums = []

    for mode in modes:
        # get the id by reading the mode file
        with open(f'../{mode}.json') as fp:
            data = json.load(fp)

        if mode == 'test':
            df_medicad = pd.read_excel('../medicad.xlsx')

        for image in data['images']:
            filename = image['file_name']
            num = get_num_from_name(filename)

            if mode == 'test':
                # now get the actual number in the dataframe:
                dfiloc = get_iloc_from_num(num, df_medicad)

                # only carry on if the number is positive
                if dfiloc > 0:
                    df_nums.append(dfnum(num))
            else:
                df_nums.append(dfnum(num))

    return df.iloc[df_nums]


def print_mean_std(df, key, keytext, suffiz=''):
    """print mean and std deviation"""
    arr = df[key]
    mean = round(np.mean(arr), 1)
    std = round(np.std(arr), 1)
    # print(f'{keytext}: {mean} ± {std} {suffiz}')
    return f'{keytext} [{suffiz}]', f'{mean} ± {std}'


def print_sum(df, loc_key='Seite', match='LI', keytext='Left'):
    """print the local sum"""
    loc_sum = np.sum([loc == match for loc in df[loc_key]])
    per = round(100 * (loc_sum / len(df)), 1)
    # print(f'{keytext}: {loc_sum} ({per} %)')
    return keytext, f'{loc_sum} ({per}%)'


def print_infos(dict_infos, modes, prename, df_len_given=0):
    """print the infos of the dataset"""

    df_loc = dataset_info(modes)
    df_len = len(df_loc)
    if df_len_given == 0:
        df_len_given = df_len

    prename = f'{prename} (n={df_len})'
    dict_infos[prename] = {}

    # print(f'{prename}: (n = {df_len}, {per}%)')
    for dfkey, dftext, suffiz in zip(DF_KEYS, DF_TEXT, DF_METRICS):
        key, value = print_mean_std(df_loc, dfkey, dftext, suffiz)
        dict_infos[prename][key] = value
    key, value = print_sum(df_loc, 'Seite', match='LI', keytext='Left')
    dict_infos[prename][key] = value
    key, value = print_sum(df_loc, 'Geschl. Pat.', match='W', keytext='Female')
    dict_infos[prename][key] = value
    # print('')
    return df_len_given, dict_infos


def print_datasets():
    """Print all dataset infromations"""
    dict_infos = {}
    all_len, dict_infos = print_infos(dict_infos, hp.SPLIT.keys(), 'overall')

    for mode in ['train', 'valid', 'test']:
        _, dict_infos = print_infos(
            dict_infos, [mode], mode, df_len_given=all_len)

    dict_infos = pd.DataFrame(dict_infos)
    print(dict_infos.to_latex())
    with pd.ExcelWriter(f'../results/analysis/dataset.xlsx') as writer:
        dict_infos.to_excel(writer)

# functions for evaluating the dataset


def perform_model_study(mode='test', model_lim=30000, model_step=5000):

    for model in range(model_step, model_lim, model_step):
        model_num = (model - 1)
        model_str = f'model_{model_num:07d}.pth'

        path = f'../results/{model_str}'

        #evaluate_a_whole_dataset(mode, extra_mode=True, model_name=model_str)
        dict_all = eval_all_angles(mode, extra_mode=True, model_name=model_str)
        visualize_all_angles(dict_all, path=path)
        eval_all_rmse(mode, path=path, model_name=model_str)


def evaluate_a_whole_dataset(mode, res_dir='../results/', extra_mode=False, model_name='model_final.pth', from_eval_model=True):
    """evaluation for a whole dataset in one func"""
    if extra_mode:
        res_dir = f'../results/{model_name}/'
    else:
        # load dataset to get the length
        res_dir = res_dir + mode

    with open(f'../{mode}.json') as fp:
        data = json.load(fp)

    all_annos_dataset = {
        'label': [],
        'predict': [],
    }

    # go trough each element
    hp.create_local_directory(res_dir)
    # use_whole_image -> subnetwork on whole image
    evaluator = Evaluator(mode=mode, angle_vis=True,
                          from_eval_model=from_eval_model,
                          use_overall_image=False,
                          ocr_analysis=False)

    for idx in tqdm(range(len(data['images']))):
        # mode = 2 -> use trainstats ; mode = 1 -> use exact detected range
        img, all_annos, all_annos_net = evaluator(
            idx=idx, optimize_area=True, mode=2)

        all_annos_dataset['label'].append(json_prep_anno(all_annos, net=False))
        all_annos_dataset['predict'].append(
            json_prep_anno(all_annos_net, net=True))
        img.save(f'{res_dir}/{idx}.png')

    with open(f'{res_dir}/{mode}_res.json', 'w') as fp:
        json.dump(all_annos_dataset, fp, indent=2)


def get_rmse(anno, anno_net, mm20=20):
    """get the rmse for all angles"""
    rmse_dict = {}
    keyp_dict = {}

    # over all categories
    for loc_cat in CATEGORIES:
        cat = loc_cat['supercategory']
        keynames = loc_cat['keypoints']

        loc_anno = anno[cat]['keypoints']
        loc_anno_net = anno_net[cat]['keypoints']

        # if min lenght available
        if len(loc_anno_net) > 2:
            rmse_arr = keyarrdiff(loc_anno, loc_anno_net, mm20=mm20)
        else:
            rmse_arr = [np.nan for _ in range(len(keynames))]

        rmse_dict[cat] = rmse_arr
        # fill the keydict
        for ind, keyname in enumerate(keynames):
            keyp_dict[keyname] = rmse_arr[ind]

    return rmse_dict, keyp_dict


def get_dice(anno, anno_net, mode='seg'):
    """get the dice for all segmentations"""
    dice_dict = {}

    # over all categories
    for cat in CATNAMES:

        if mode == 'seg':

            if 'segmentation' in anno[cat]:
                loc_anno = anno[cat]['segmentation']

                if 'segmentation' in anno_net[cat]:
                    loc_anno_net = anno_net[cat]['segmentation']
                else:
                    loc_anno_net = [[0, 0, 1, 1]]

                dice_dict[cat] = dice_seg(loc_anno, loc_anno_net)
            else:
                dice_dict[cat] = np.nan

        else:
            loc_anno = anno[cat]['bbox']
            loc_anno_net = anno_net[cat]['bbox']

            if len(loc_anno_net) < 4:
                loc_anno_net = [0, 0, 1, 1]

            dice_dict[cat] = dice_bb(loc_anno, loc_anno_net)

    return dice_dict


def poly_2_mask(polys):
    img = Image.new("L", (3500, 13000), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        draw.polygon((poly), outline=1, fill=1)
    mask = np.array(img)
    return mask


def dice_seg(seg1, seg2):
    """compute the dice for two segmentations"""

    # create mask from segmentation
    seg1 = poly_2_mask(seg1)
    seg2 = poly_2_mask(seg2)

    seg1 = np.array(seg1)
    seg2 = np.array(seg2)

    # get the intersection
    intersection = np.sum(seg1 * seg2)

    # get the union
    union = np.sum(seg1) + np.sum(seg2)

    # compute the dice
    dice = (2 * intersection) / union

    return dice


def check_overlap(bb1, bb2):
    """check if two bounding boxes overlap"""
    # get the intersection
    xmin1, ymin1, width1, height1 = bb1
    xmin2, ymin2, width2, height2 = bb2

    xmax1 = xmin1 + width1
    ymax1 = ymin1 + height1

    xmax2 = xmin2 + width2
    ymax2 = ymin2 + height2

    # check if the two boxes overlap
    if xmin1 < xmax2 and xmax1 > xmin2 and ymin1 < ymax2 and ymax1 > ymin2:
        return True
    else:
        return False


def dice_bb(bb1, bb2):
    """compute the dice for two bounding boxes
    bbox = [xmin, ymin, width, height]
    """
    if not check_overlap(bb1, bb2):
        return 0

    xmin1, ymin1, width1, height1 = bb1
    xmin2, ymin2, width2, height2 = bb2

    bb1_area = width1 * height1
    bb2_area = width2 * height2

    overlap_bb = [max(xmin1, xmin2), max(ymin1, ymin2), min(
        xmin1 + width1, xmin2 + width2), min(ymin1 + height1, ymin2 + height2)]
    intersection = (overlap_bb[2] - overlap_bb[0]) * \
        (overlap_bb[3] - overlap_bb[1])

    # get the union
    union = bb1_area + bb2_area

    # compute the dice
    dice = (2 * intersection) / union

    return dice


def get_20mm(anno):
    """the points K1 and K2 are 20mm apart"""
    kparr = anno['K']['keypoints']
    K1 = np.array([kparr[0], kparr[1]])
    K2 = np.array([kparr[3], kparr[4]])
    return np.linalg.norm(K1-K2)


def eval_all_rmse(mode='test', path='../results/analysis', model_name='', digits=2, do_dice=True):
    """get_all rmse"""
    key_list = [18, 80, 91, 111, 116, 127]

    if len(model_name) > 1:
        with open(f'../results/{model_name}/{mode}_res.json') as fp:
            data = json.load(fp)
    else:
        with open(f'../results/{mode}/{mode}_res.json') as fp:
            data = json.load(fp)

    rmse_means = {
        cat: [] for cat in CATNAMES
    }

    dice_means_bb = {
        cat: [] for cat in CATNAMES
    }

    dice_means_seg = {
        cat: [] for cat in CATNAMES
    }

    all_keyp_dict = {}

    # create a dict with all the keypoint names
    for loc_cat in CATEGORIES:
        keynames = loc_cat['keypoints']

        for keyname in keynames:
            all_keyp_dict[keyname] = []

    # evaluate whole dataset
    for loc_index, (anno, anno_net) in tqdm(enumerate(zip(data['label'], data['predict']))):
        if loc_index in key_list:
            print(loc_index)
            continue
        # get 20 mm distance
        mm20 = get_20mm(anno)
        # evaluate keypoint, dicebb and dice_seg acuracy
        rmse_dict, keyp_dict = get_rmse(anno, anno_net, mm20)

        if do_dice:
            dice_dict_bb = get_dice(anno, anno_net, mode='bb')
            dice_dict_seg = get_dice(anno, anno_net, mode='seg')

        for cat in rmse_dict.keys():
            rmse_means[cat].append(np.nanmean(rmse_dict[cat]))

            if do_dice:
                dice_means_bb[cat].append(dice_dict_bb[cat])
                dice_means_seg[cat].append(dice_dict_seg[cat])

        for keyname in keyp_dict.keys():
            all_keyp_dict[keyname].append(keyp_dict[keyname])

    keyp_res = {}
    for keyname in keyp_dict.keys():
        mean = np.nanmean(all_keyp_dict[keyname])
        std = np.nanstd(all_keyp_dict[keyname])
        keyp_res[keyname] = f'{mean:.{digits}f} ± {std:.{digits}f}'

    dict_res = {}
    all_res_rmse = []
    all_res_dice_bb = []
    all_res_dice_seg = []

    # finally print all results
    for cat in CATNAMES:
        all_res_rmse.extend(rmse_means[cat])
        mean = round(np.nanmean(rmse_means[cat]), digits)
        std = round(np.nanstd(rmse_means[cat]), digits)
        dict_res[f'{cat}'] = [f'{mean} ± {std}']

        if do_dice:
            all_res_dice_bb.extend(dice_means_bb[cat])
            mean = round(np.nanmean(dice_means_bb[cat]), digits)
            std = round(np.nanstd(dice_means_bb[cat]), digits)
            dict_res[f'{cat}'].append(f'{mean} ± {std}')

            all_res_dice_seg.extend(dice_means_seg[cat])
            mean = round(np.nanmean(dice_means_seg[cat]), digits)
            std = round(np.nanstd(dice_means_seg[cat]), digits)
            dict_res[f'{cat}'].append(f'{mean} ± {std}')

    mean = round(np.nanmean(all_res_rmse), digits)
    std = round(np.nanstd(all_res_rmse), digits)
    dict_res['All'] = [f'{mean} ± {std}']

    if do_dice:
        mean = round(np.nanmean(all_res_dice_bb), digits)
        std = round(np.nanstd(all_res_dice_bb), digits)
        dict_res['All'].append(f'{mean} ± {std}')

        mean = round(np.nanmean(all_res_dice_seg), digits)
        std = round(np.nanstd(all_res_dice_seg), digits)
        dict_res['All'].append(f'{mean} ± {std}')

    dict_res = pd.DataFrame(dict_res).T
    print(dict_res.to_latex())

    with pd.ExcelWriter(f'{path}/rmse.xlsx') as writer:
        dict_res.to_excel(writer)

    keyp_res = {}
    for keyname in all_keyp_dict.keys():
        mean = np.nanmean(all_keyp_dict[keyname])
        std = np.nanstd(all_keyp_dict[keyname])
        keyp_res[keyname] = f'{mean:.{digits}f} ± {std:.{digits}f}'

    df = pd.DataFrame(keyp_res, index=['mean']).T
    df.sort_index(inplace=True)
    print(df.to_latex())

    # create new dict with all points split by their number
    key_res_detail = {}
    for loc_cat in CATEGORIES:
        latex_name = loc_cat['supercategory']
        key_res_detail[latex_name] = ['-' for _ in range(6)]

        for ind, keyp in enumerate(loc_cat['keypoints']):
            key_res_detail[latex_name][ind] = keyp_res[keyp]

    df = pd.DataFrame(key_res_detail).T
    print(df.to_latex())

    return dict_res, all_keyp_dict


def keyarrdiff(loc_anno, loc_anno_net, mm20=20):
    x_arr_anno = loc_anno[::3]
    y_arr_anno = loc_anno[1::3]

    x_arr_anno_net = loc_anno_net[::3]
    y_arr_anno_net = loc_anno_net[1::3]

    diff = []
    for (x, y, xnet, ynet) in zip(x_arr_anno, y_arr_anno, x_arr_anno_net, y_arr_anno_net):
        loc_diff = [x-xnet, y-ynet]
        diff.append(np.linalg.norm(loc_diff) * (20 / mm20))

    return diff


def get_iloc_from_num(num, df):
    """get the location of the current id in the test df excel"""
    for idx in range(len(df)):
        df_loc = df.iloc[idx]
        patid = int(df_loc['Pat. Nummer'])
        varval = df_loc['Var/val']
        if (patid == num) and (type(varval) == str):
            return idx
    return -1


def eval_all_angles(mode='test', excel=True, inter=False, rmse=False, extra_mode=False, model_name='model_final.pth'):
    """get_all rmse"""

    key_list = [18, 80, 91, 111, 116, 127]

    if extra_mode:
        json_name = f'../results/{model_name}/{mode}_res.json'
        save_dir = f'../results/{model_name}/'
    else:
        save_dir = '../results/analysis'
        json_name = f'../results/{mode}/{mode}_res.json'

    with open(json_name) as fp:
        data = json.load(fp)

    with open(f'../{mode}.json') as fp:
        data_origin = json.load(fp)

    angle_errors = {
        angle: [] for angle in ANGLE_NAMES
    }

    df = pd.read_excel('../medicad.xlsx')

    if USE_CLINIC_EXCEL:
        df2 = pd.read_excel('../osteo.xlsx')

    dict_all = {}
    dict_all['index'] = []
    for angle in ANGLE_NAMES:
        for loc_rater in RATERS:
            dict_all[f'{angle}_{loc_rater}'] = []

    nums = []

    for index, (anno, anno_net, image) in tqdm(enumerate(zip(data['label'], data['predict'], data_origin['images']))):

        if index in key_list:
            print(index)
            continue

        carry_on = True
        mikolicz_percentage = 55

        num = index
        if excel:
            filename = image['file_name']
            num = get_num_from_name(filename)

            # now get the actual number in the dataframe:
            dfiloc = get_iloc_from_num(num, df)

            # only carry on if the number is positive
            if dfiloc > 0:
                df_res = df.iloc[dfiloc]
                mikolicz_percentage = df_res['% Korrektur']

                if USE_CLINIC_EXCEL:
                    df_res_clinic = df2.iloc[num-1]
            else:
                carry_on = False

        if carry_on:
            nums.append(num)
            dict_all['index'].append(index)

            overall_img = cv2.imread(image['file_name'])

            # Original Annotation
            anno_res = anno_calculation(
                overall_img, anno, option=2, acc_boost=False, k_len=25)

            anno_net_res = anno_calculation(
                overall_img, anno_net, option=2, acc_boost=True, k_len=25)

            for angle in ANGLE_NAMES:

                dict_all[f'{angle}_OS1'].append(round(anno_res[angle], 1))
                dict_all[f'{angle}_AI'].append(round(anno_net_res[angle], 1))
                err = abs(round(anno_res[angle], 1) -
                          round(anno_net_res[angle], 1))

                if excel:
                    try:
                        excel_angle = float(str(df_res[angle]).replace(
                            '°', '').replace(',', '.'))
                        dict_all[f'{angle}_OS2'].append(round(excel_angle, 1))
                        err = abs(0.5*(excel_angle + anno_res[angle]) -
                                  anno_net_res[angle])
                    except KeyError:
                        excel_angle = anno_res[angle]
                        dict_all[f'{angle}_OS2'].append(np.nan)

                    if USE_CLINIC_EXCEL:
                        try:
                            excel_angle2 = float(str(df_res_clinic[angle]).replace(
                                '°', '').replace(',', '.'))
                            
                            if angle == 'JLCA':
                                excel_angle2 = abs(excel_angle2)

                            dict_all[f'{angle}_OS3'].append(
                                round(excel_angle2, 1))
                            if not math.isnan(excel_angle2):

                                err = abs(1/3*(excel_angle + anno_res[angle] + excel_angle2) -
                                          anno_net_res[angle])
                        except:
                            dict_all[f'{angle}_OS3'].append(np.nan)
                else:
                    err = abs(anno_res[angle] - anno_net_res[angle])

                if inter:
                    err = abs(df_res[angle] - anno_res[angle])

                angle_errors[angle].append(err)

    all_res = []
    # finally print all results
    print('\n')
    res_dict = {}
    for angle, unit in zip(ANGLE_NAMES, UNITS):
        all_res.extend(angle_errors[angle])

        mean = round(np.nanmean(angle_errors[angle]), 1)
        std = round(np.nanstd(angle_errors[angle]), 1)
        if rmse:
            sq_err = [ang**2 for ang in angle_errors[angle]]
            mean = round(np.sqrt(np.nanmean(sq_err)), 1)

        ang = angle.replace('\n(°)', '')
        #print(f'{ang}: {mean} ± {std}')
        res_dict[f'{ang} [{unit}]'] = [f'{mean} ± {std}']

    mean = round(np.nanmean(all_res), 1)
    std = round(np.nanstd(all_res), 1)
    if rmse:
        sq_err = [ang**2 for ang in all_res]
        mean = round(np.sqrt(np.nanmean(sq_err)), 1)

    #print(f'All: {mean} ± {std}')
    res_dict['All'] = [f'{mean} ± {std}']
    # print(res_dict)
    res_dict = pd.DataFrame(res_dict).T

    with pd.ExcelWriter(f'{save_dir}angles.xlsx') as writer:
        res_dict.to_excel(writer)

    print(res_dict.to_latex())

    all_angles = pd.DataFrame.from_dict(dict_all)

    if USE_EXCEL:
        all_angles.insert(0, "Pat-ID", nums)
        all_angles = all_angles.set_index('Pat-ID')
        all_angles = all_angles.sort_index()

    all_angles.to_excel(f'{save_dir}res.xlsx')

    return dict_all


def visualize_all_angles(dict_all, path='../results/analysis'):
    """visualize the angles from the dictionary"""

    for index, angle in enumerate(ANGLE_NAMES):
        # prepare plot
        fig = plt.figure(figsize=(28, 12))
        loc_angle_str = ANGLE_NAMES[index].split('\n')[0]
        fig.suptitle(loc_angle_str, fontsize=20)

        # get all possible combinations
        all_comps = get_all_combs(RATERS)
        len_all_comps = len(all_comps)

        for loc_count, comp in enumerate(all_comps):
            key_loc1 = f'{angle}_{comp[0]}'
            key_loc2 = f'{angle}_{comp[1]}'

            arr_loc1 = np.array(dict_all[key_loc1])
            arr_loc2 = np.array(dict_all[key_loc2])

            # bland altman plottting
            bland_altman_plot(comp, arr_loc1, arr_loc2, index,
                              loc_count+1, maxcount=len_all_comps)

        # adjust axis and save
        if len_all_comps > 1:
            reset_axes(max_plots=len_all_comps)
        fig.savefig(f'{path}/{loc_angle_str}.png')


def reset_axes(max_plots=3):
    """reset the axis limits to allow direct comparison"""
    ymin = 1e6
    ymax = -1e6

    xmin = 1e6
    xmax = -1e6

    for i in range(1, max_plots + 1):
        plt.subplot(1, max_plots, i)

        ymin_loc, ymax_loc = plt.ylim()
        xmin_loc, xmax_loc = plt.xlim()

        if ymin_loc < ymin:
            ymin = ymin_loc

        if ymax_loc > ymax:
            ymax = ymax_loc

        if xmin_loc < xmin:
            xmin = xmin_loc

        if xmax_loc > xmax:
            xmax = xmax_loc

    for i in range(1, max_plots + 1):
        plt.subplot(1, max_plots, i)
        plt.ylim((ymin, ymax))
        plt.xlim((xmin, xmax))


def bland_altman_plot(naming, data1, data2, index, count, fs=16, maxcount=3):
    """perform the bland_altman plot between the two arrays"""

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.nanmean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.nanmean(diff)                   # Mean of the difference
    # Standard deviation of the difference
    sd = np.nanstd(diff, axis=0)

    ax = plt.subplot(1, maxcount, count)

    # title
    title_str = f'{naming[0]} vs. {naming[1]}'
    plt.title(title_str, fontsize=fs+2)

    plt.scatter(mean, diff)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

    # labels
    if count == 1:
        loc_angle_str = ANGLE_NAMES[index].split('\n')[0]
        plt.ylabel(
            f'Difference of Measured {loc_angle_str} [{UNITS[index]}]', fontsize=fs)
    plt.xlabel(f'Mean of Measurements [{UNITS[index]}]', fontsize=fs)

    # add a textbox
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        r'+1.96 SD: $%.2f$' % (md + 1.96*sd, ),
        r'mean diff: $%.2f$' % (md, ),
        r'-1.96 SD: $%.2f$' % (md - 1.96*sd, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def exclude_outliers(dict_all, fac=2, mode='test', cats=ANGLE_NAMES):
    """exclude test images with a std deviation of over {3}"""
    excluded_indices = []

    for angle in cats:
        key_anno = f'{angle}_anno'
        key_net = f'{angle}_net'

        anno_arr = np.array(dict_all[key_anno])
        net_arr = np.array(dict_all[key_net])
        indexes = dict_all['index']

        data1 = np.asarray(anno_arr)
        data2 = np.asarray(net_arr)
        diff = data1 - data2
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        for loc_ind, loc_diff in zip(indexes, diff):

            if loc_diff > md + fac*sd:
                excluded_indices.append(loc_ind)

            if loc_diff < md - fac*sd:
                excluded_indices.append(loc_ind)

    excluded_indices = list(set(excluded_indices))
    print(excluded_indices)

    with open(f'../{mode}.json') as fp:
        data = json.load(fp)
    excluded_filenames = [data['images'][idx]['file_name']
                          for idx in excluded_indices]

    print(excluded_filenames)

    for angle in cats:
        for loc_str in RATERS:
            key = f'{angle}_{loc_str}'
            dict_all[key] = exclude_by_indices(dict_all[key], excluded_indices)

    return dict_all


def exclude_by_indices(data, excluded_indices):
    normal_indices = list(range(len(data)))
    excluded = [
        index for index in normal_indices if index not in excluded_indices]
    data = [data[i] for i in excluded]
    return data


def irr(dict_all, digits=1, tol=0.1, xlsx_name='irr'):
    """Calculate the Inter Rater Reliability"""
    # go trough all possibilities of rater combinations
    irr = {}
    all_comps = get_all_combs(RATERS)
    # go trough all angles
    for angle in ANGLE_NAMES:
        irr[angle] = {}

        for comp in all_comps:
            comp_name = f'{comp[0]}_{comp[1]}'
            arr0_name = f'{angle}_{comp[0]}'
            arr1_name = f'{angle}_{comp[1]}'

            arr0 = [round(ele, digits) for ele in dict_all[arr0_name]]
            arr1 = [round(ele, digits) for ele in dict_all[arr1_name]]

            score = sum([1 if abs(ele0 - ele1) <= tol else 0 for ele0,
                         ele1 in zip(arr0, arr1)]) / len(arr0)
            score = round(score * 100)

            irr[angle][comp_name] = score

    irr = pd.DataFrame(irr).T

    with pd.ExcelWriter(f'../results/analysis/{xlsx_name}.xlsx') as writer:
        irr.to_excel(writer, sheet_name=f'tol={tol}')

    print(irr.to_latex())

    return irr


def get_all_combs(arr):
    """
    combine all elements of arr with one another
    # exp: [excel, anno, net]
    # => [[excel, anno], [excel, net], ...]
    """
    len_a = len(arr)
    all_comps = []
    # go tough all elements
    for i, ele1 in enumerate(arr):
        if i + 1 < len_a:
            # go trough all elements following the current element
            for ele2 in arr[i+1:]:
                loc_arr = [ele1, ele2]
                all_comps.append(loc_arr)
    return all_comps


def produce_multiple_iccs(dict_all, all_key='all'):
    icc_res = {}
    combs = get_all_combs(RATERS)

    for comb in combs:
        loc_name = f'{comb[0]}_{comb[1]}'
        icc_res[loc_name] = intra_class_correlation(
            dict_all, xlsx_name=f'ICC_{comb[0]}_{comb[1]}', compare=comb)

    icc_res = pd.DataFrame(icc_res)
    with pd.ExcelWriter(f'../results/analysis/icc.xlsx') as writer:
        icc_res.to_excel(writer)

    print(icc_res.to_latex())


def intra_class_correlation(dict_all, xlsx_name='ICC_all', compare=['net', 'anno', 'excel'], loc_row=2):
    """produce the intra_correlation"""
    icc_res = {}
    with pd.ExcelWriter(f'../results/analysis/{xlsx_name}.xlsx') as writer:
        for cat in ANGLE_NAMES:

            em_dict = {
                'ID': [],
                'Rater': [],
                'Scores': []
            }

            all_keys = dict_all.keys()

            for key in all_keys:
                corresponding_cat = key.split('_')[0]

                # check if key contains relevant rater and rel names
                if cat == corresponding_cat and any(comp in key for comp in compare):

                    em_dict['Scores'].extend(dict_all[key])

                    em_dict['ID'].extend(list(range(len(dict_all[key]))))

                    em_dict['Rater'].extend([key.replace(cat, '')
                                             for _ in dict_all[key]])

            data = pd.DataFrame(em_dict)

            try:
                icc = pg.intraclass_corr(data=data, targets='ID', raters='Rater',
                                         ratings='Scores', nan_policy='omit')
                icc.to_excel(writer, sheet_name=cat)

                val = round(icc.iloc[loc_row]['ICC'], 2)
                ci95 = icc.iloc[loc_row]['CI95%']
                icc_res[cat] = f'{val} [{ci95[0]}, {ci95[1]}]'
            except AssertionError:
                icc_res[cat] = f'{np.nan} [{np.nan}, {np.nan}]'

    return icc_res


def perform_all_tests(mode='test'):
    """perform the complete analysis on the dataset"""
    # perform the deep learning calculations
    evaluate_a_whole_dataset(mode)

    print('ANGLES:')
    dict_all = eval_all_angles(mode=mode, inter=False, excel=USE_EXCEL)

    # exclude outliers
    dict_all = exclude_outliers(dict_all)

    # visualize
    visualize_all_angles(dict_all)

    # gt the rmse of points
    print('\nRMSE')
    eval_all_rmse(mode='test')

    # get the inter reader reliability depending on the points
    print('\nInter Reader Reliability (tol = 1°)')
    irr(dict_all, tol=1)

    print('\nIntraclass Correlation')
    produce_multiple_iccs(dict_all)


def icc_model_gt(dict_all, use_all=False):
    """define the icc correlation between the ai model and the ground truth"""

    # first extend the dictionary with the gt
    use_raters = RATERS if use_all else RATERS[:-1]
    iterate = len(dict_all[list(dict_all.keys())[0]])

    for angle in ANGLE_NAMES:
        angle_arr = []

        # get all errors
        for i in range(iterate):
            loc_array = []

            for rater in use_raters:
                loc_array.append(dict_all[f'{angle}_{rater}'][i])

            loc_value = np.nanmean(loc_array)
            angle_arr.append(loc_value)

        dict_all[f'{angle}_gt'] = angle_arr

    comb = [RATERS[len(RATERS) - 1], 'gt']

    icc_res = intra_class_correlation(
        dict_all, xlsx_name=f'ICC_{comb[0]}_{comb[1]}', compare=comb)

    for key in icc_res.keys():
        icc_res[key] = [icc_res[key]]

    icc_res = pd.DataFrame(icc_res).T
    print(icc_res.to_latex())
    return icc_res


# %%
if __name__ == '__main__':
    # perform_model_study()

    print('DATASET:')
    print_datasets()
    # evaluate_a_whole_dataset(MODE, from_eval_model=True)

    print('ANGLES:')
    dict_all = eval_all_angles(mode=MODE, inter=False, excel=USE_EXCEL)

    # save dict_all as intern dict
    with open('../results/analysis/dict_all_int.json', 'w') as f:
        json.dump(dict_all, f, indent=4)

    # get the rmse of points
    print('\nRMSE')
    dict_res, all_keyp_dict = eval_all_rmse(mode=MODE, do_dice=True)

    # visualize
    visualize_all_angles(dict_all)

    # get the inter-reader-reliability depending on the points
    print('\nInter Reader Reliability')
    irr(dict_all, tol=1)

    print('\nIntraclass Correlation')
    produce_multiple_iccs(dict_all)

    # display outliers
    icc_model_gt(dict_all, use_all=False)

    all_angles = pd.DataFrame.from_dict(dict_all)
    all_angles.to_excel(f'test_internal_all.xlsx')
    produce_multiple_iccs(dict_all)


