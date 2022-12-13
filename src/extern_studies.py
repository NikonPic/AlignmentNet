# %%
from cmath import nan
from string import digits
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
from ipywidgets import widgets
import os
from sympy import Rel
from tqdm import tqdm
import pingouin as pg
from angle_calc import anno_calculation, json_prep_anno
from categories import CATNAMES
from eval_angle_test import get_all_combs, reset_axes
from train_detectron import Evaluator
import datetime

# from eval_angle_test import dict_all_intern

PATH_EXT = '../images_external'
SAVE_DIR = '../results/test_ext'
FILES_EXT = os.listdir(PATH_EXT)
FILES_EXT = [f'{PATH_EXT}/{f}' for f in FILES_EXT]

# contains angle, unit and tolerance
REL_KEY_DICT = {
    'Mikulicz Länge  präOP': ['Mikulicz', 'mm', 5],
    'M auf TP präOP': ['Mikulicz auf TP', '%', 2],
    'MAD präOP': ['MAD', 'mm', 2],
    'mLPFA präOP': ['mLPFA', '°', 2],
    'AMA präOP': ['AMA', '°', 2],
    'mLDFA präOP': ['mLDFA', '°', 2],
    'JLCA präOP': ['JLCA', '°', 2],
    'mMPTA präOP': ['mMPTA', '°', 2],
    'mFA- mTA präoOP': ['mFTA', '°', 2],
    'KJLO präOP': ['KJLO', '°', 2],
    'mLDTA präOP': ['mLDTA', '°', 2],
}
ANGLE_LIST = [ang[0] for ang in REL_KEY_DICT.values()]
VALUE_LIST = [ang[1] for ang in REL_KEY_DICT.values()]
RATERS = ['OS1', 'OS2', 'OS3', 'AI']
SW_RATERS = ['AI', 'OS1', 'OS2', 'OS3']

CUT_LEN = 'Korrektur (mm)'
CUT_ANGLE = 'Winkel'


def preview_external():
    all_annos_dataset = {}
    data = pd.read_excel('../extern_data.xlsx')
    filenames = data['Pat. Nummer']

    for ind, f in tqdm(enumerate(filenames)):
        evaluator = Evaluator(onlyimage=False, angle_vis=True,
                              truelabel=False, from_eval_model=True,
                              ext=True)

        filename = f'../images_external/{f}.png'
        img, all_annos_net = evaluator.forward_raw(filename)
        all_annos_dataset[f'{f}'] = json_prep_anno(all_annos_net, net=True)
        img.save(f'../results/test_ext/{ind}_{f}.png')

    with open(f'../results/test_ext/res.json', 'w') as fp:
        json.dump(all_annos_dataset, fp, indent=2)


def update_ext(idx=77):
    data = pd.read_excel('../extern_data.xlsx')
    filenames = data['Pat. Nummer']
    filename = f'../images_external/{filenames[idx]}.png'
    evaluator = Evaluator(onlyimage=True, angle_vis=True,
                          truelabel=False, from_eval_model=True, ext=True)
    return evaluator.forward_raw(filename)


def build_dict_all_from_ext_dataset():
    """build a dictionary with all patients and their angles"""

    # open excel data
    data = pd.read_excel('../extern_data.xlsx')
    filenames = data['Pat. Nummer']

    # open ai annotations
    with open(f'../results/test_ext/res.json') as fp:
        all_annos = json.load(fp)

    # dict_all = {
    #   'Mikulicz_OS1': [...],
    #   'Mikulicz_OS2': [...],
    #   ...
    # }

    dict_all = {}
    for angle_key in REL_KEY_DICT.keys():
        angle = REL_KEY_DICT[angle_key][0]
        for loc_rater in RATERS:
            dict_all[f'{angle}_{loc_rater}'] = []

    for pat_num in tqdm(filenames):
        key_append = ''

        index_num = list(filenames).index(pat_num)

        # add all angles from the excel
        for rat_ind, loc_rater in enumerate(RATERS[:-1]):

            if rat_ind > 0:
                key_append = f'.{rat_ind}'

            # iterate over all angles
            for angle_key in REL_KEY_DICT.keys():
                angle_name = REL_KEY_DICT[angle_key][0]

                # get the key in the excel
                data_key = f'{angle_key}{key_append}'
                loc_data_list = list(data[data_key])
                # take the angle
                angle = loc_data_list[index_num]
                # format the angle
                try:
                    angle = float(angle)
                    angle = round(angle, 1)
                except ValueError:
                    angle = np.nan

                # append to the dict
                dict_all[f'{angle_name}_{loc_rater}'].append(angle)

        # add the AI
        rater = RATERS[len(RATERS)-1]
        anno = all_annos[f'{pat_num}']

        img_name = f'{PATH_EXT}/{pat_num}.png'
        overall_img = cv2.imread(img_name)
        anno_res = anno_calculation(
            overall_img, anno, option=2, acc_boost=False, k_len=20)

        for angle_key in REL_KEY_DICT.keys():
            angle_name = REL_KEY_DICT[angle_key][0]
            dict_all[f'{angle_name}_{rater}'].append(
                round(anno_res[angle_name], 1))

    all_angles = pd.DataFrame.from_dict(dict_all)
    all_angles.to_excel(f'{SAVE_DIR}/res.xlsx')

    return dict_all


def acc_study(dict_all, use_all=False, digits=2):
    """perform an accuracy study"""
    rater_error = {}
    for rater in RATERS:
        angle_errors = {
            angle: [] for angle in ANGLE_LIST
        }

        iterate = len(dict_all[list(dict_all.keys())[0]])

        # get all errors
        for i in range(iterate):
            for angle in ANGLE_LIST:
                angle_raters = []

                rater_arr = RATERS if use_all else RATERS[:-1]
                for loc_rater in rater_arr:
                    angle_raters.append(dict_all[f'{angle}_{loc_rater}'][i])

                mean_raters = np.nanmean(angle_raters, axis=0)

                ai_value = dict_all[f'{angle}_{rater}'][i]
                err = abs(mean_raters - ai_value)

                angle_errors[angle].append(err)

        rater_error[rater] = angle_errors

    all_errors = {}
    for rater in rater_error.keys():
        rater_means = []
        rater_std = []
        all_errors[rater] = {}
        for angle in rater_error[rater].keys():
            mean_err = round(np.nanmean(rater_error[rater][angle]), digits)
            std_err = round(np.nanstd(rater_error[rater][angle]), digits)
            rater_means.append(mean_err)
            rater_std.append(std_err)
            all_errors[rater][angle] = f'{mean_err} ± {std_err}'

        mean_all = round(np.nanmean(rater_means), digits)
        std_all = round(np.nanstd(rater_std), digits)
        all_errors[rater]['all'] = f'{mean_all} ± {std_all}'

    all_errors_df = pd.DataFrame(all_errors)
    format_table_latex(all_errors_df, max1min0=0)

    return all_errors


def new_acc_study(dict_all, digits=2):
    """take: diff os vs os and mean os vs ai"""

    combs = get_all_combs(RATERS[:-1])
    iterate = len(dict_all[list(dict_all.keys())[0]])
    rater_error = {}
    print(combs)

    # do all combinations
    for comb in combs:
        combname = f'{comb[0]}-{comb[1]}'
        rater_error[combname] = {}
        # do all angles
        for angle in ANGLE_LIST:
            angle_res = []
            # go trough dataset
            for i in range(iterate):
                # both raters
                rater1 = comb[0]
                rater2 = comb[1]

                # get the values
                rater1_value = dict_all[f'{angle}_{rater1}'][i]
                rater2_value = dict_all[f'{angle}_{rater2}'][i]

                # save the abs difference
                diff = abs(rater1_value - rater2_value)
                angle_res.append(diff)

            # get the mean and std
            mean_err = round(np.nanmean(angle_res), digits)
            std_err = round(np.nanstd(angle_res), digits)
            rater_error[combname][angle] = f'{mean_err} ± {std_err}'

    # use the old study:
    all_errors = acc_study(dict_all, use_all=False, digits=digits)
    ai_err = all_errors['AI']
    ai_err.pop('all')

    rater_name = f'{RATERS[:-1]}'
    rater_name = rater_name.replace('[', '').replace(
        ']', '').replace("'", '').replace(' ', '')
    # append the ai error
    rater_error[f'{rater_name}-{RATERS[-1]}'] = ai_err

    rater_error = pd.DataFrame(rater_error)
    format_table_latex(rater_error, max1min0=0)
    return rater_error


def format_table_latex(table, max1min0=0):
    """redesign the table to be in latex format"""

    call = np.nanargmax if max1min0 else np.nanargmin

    for index in range(len(table)):
        try:
            minindex = call(([float(val.split(' ')[0])
                              for val in table.iloc[index]]))
        except AttributeError:
            minindex = call(([float(val) for val in table.iloc[index]]))

        minval = table.iloc[index][minindex]
        minval = f'{minval}'
        table.iloc[index, minindex] = str('\\' + 'textbf{' + minval + '}')

    string_res = table.to_latex()
    string_res = string_res.replace('\\textbackslash ', '\\').replace(
        '\\{', '{').replace('\\}', '}')
    print(string_res)


def angle_acc(dict_all, rmse=False, use_all=False):
    """calcuate the angle accuracy"""
    angle_errors = {
        angle: [] for angle in ANGLE_LIST
    }

    iterate = len(dict_all[list(dict_all.keys())[0]])

    # get all errors
    for i in tqdm(range(iterate)):
        for angle in ANGLE_LIST:
            angle_raters = []

            rater_arr = RATERS if use_all else RATERS[:-1]
            for rater in rater_arr:
                angle_raters.append(dict_all[f'{angle}_{rater}'][i])

            mean_raters = np.nanmean(angle_raters, axis=0)
            ai_value = dict_all[f'{angle}_AI'][i]
            err = abs(mean_raters - ai_value)

            angle_errors[angle].append(err)

    all_res = []
    # finally print all results
    print('\n')
    res_dict = {}
    for loc_key in REL_KEY_DICT.keys():
        angle, unit, _ = REL_KEY_DICT[loc_key]
        all_res.extend(angle_errors[angle])

        mean = round(np.mean(angle_errors[angle]), 1)
        std = round(np.std(angle_errors[angle]), 1)

        if rmse:
            sq_err = [ang**2 for ang in angle_errors[angle]]
            mean = round(np.sqrt(np.mean(sq_err)), 1)

        ang = angle.replace('\n(°)', '')
        res_dict[f'{ang} [{unit}]'] = [f'{mean} ± {std}']

    mean = round(np.mean(all_res), 1)
    std = round(np.std(all_res), 1)
    if rmse:
        sq_err = [ang**2 for ang in all_res]
        mean = round(np.sqrt(np.mean(sq_err)), 1)

    res_dict['All'] = [f'{mean} ± {std}']
    res_dict = pd.DataFrame(res_dict).T

    with pd.ExcelWriter(f'{SAVE_DIR}angles.xlsx') as writer:
        res_dict.to_excel(writer)

    print(res_dict.to_latex())


def visualize_all_angles(dict_all, path='../results/analysis_ext'):
    """visualize the angles from the dictionary"""

    for index, angle in enumerate(ANGLE_LIST):
        # prepare plot
        fig = plt.figure(figsize=(24, 12))
        loc_angle_str = ANGLE_LIST[index].split('\n')[0]
        fig.suptitle(loc_angle_str, fontsize=20)

        # get all possible combinations
        all_comps = get_all_combs(SW_RATERS)
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
        loc_angle_str = ANGLE_LIST[index].split('\n')[0]
        plt.ylabel(
            f'Difference of Measured {loc_angle_str} [{VALUE_LIST[index]}]', fontsize=fs)
    plt.xlabel(f'Mean in [{VALUE_LIST[index]}]', fontsize=fs)

    # add a textbox
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((
        r'+1.96 SD: $%.2f$' % (md + 1.96*sd, ),
        r'mean diff: $%.2f$' % (md, ),
        r'-1.96 SD: $%.2f$' % (md - 1.96*sd, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def irr(dict_all, digits=2, tol=0.1, xlsx_name='irr', include_ai=True):
    """Calculate the Inter Rater Reliability"""
    # go trough all possibilities of rater combinations
    irr = {}
    if include_ai:
        all_comps = get_all_combs(RATERS[:-1])
    else:
        all_comps = get_all_combs(RATERS)
    # go trough all angles
    for key in REL_KEY_DICT.keys():
        angle, unit, tol = REL_KEY_DICT[key]

        dict_name = f'{angle} (tol={tol}{unit})'
        irr[dict_name] = {}

        for comp in all_comps:
            comp_name = f'{comp[0]}_{comp[1]}'
            arr0_name = f'{angle}_{comp[0]}'
            arr1_name = f'{angle}_{comp[1]}'

            arr0 = [round(ele, digits) for ele in dict_all[arr0_name]]
            arr1 = [round(ele, digits) for ele in dict_all[arr1_name]]

            score = sum([1 if abs(ele0 - ele1) <= tol else 0 for ele0,
                         ele1 in zip(arr0, arr1)]) / len(arr0)
            score = round(score * 100, digits)

            irr[dict_name][comp_name] = score

    # compare with gt
    if include_ai:
        for key in REL_KEY_DICT.keys():
            angle, unit, tol = REL_KEY_DICT[key]

            dict_name = f'{angle} (tol={tol}{unit})'

            comp = ['gt', 'AI']

            comp_name = f'{comp[0]}_{comp[1]}'
            arr0_name = f'{angle}_{comp[0]}'
            arr1_name = f'{angle}_{comp[1]}'

            arr0 = [round(ele, digits) for ele in dict_all[arr0_name]]
            arr1 = [round(ele, digits) for ele in dict_all[arr1_name]]

            score = sum([1 if abs(ele0 - ele1) <= tol else 0 for ele0,
                         ele1 in zip(arr0, arr1)]) / len(arr0)
            score = round(score * 100, digits)

            irr[dict_name][comp_name] = score

    irr = pd.DataFrame(irr).T

    with pd.ExcelWriter(f'../results/analysis_ext/{xlsx_name}.xlsx') as writer:
        irr.to_excel(writer, sheet_name=f'tol={tol}')

    # format_table_latex(irr, max1min0=1)

    return irr


def irr_gt(dict_all, digits=1):
    irr = {}
    for key in REL_KEY_DICT.keys():
        angle, unit, tol = REL_KEY_DICT[key]

        dict_name = f'{angle} (tol={tol}{unit})'
        irr[dict_name] = {}

        comp = ['gt', 'AI']

        comp_name = f'{comp[0]}_{comp[1]}'
        arr0_name = f'{angle}_{comp[0]}'
        arr1_name = f'{angle}_{comp[1]}'

        arr0 = [round(ele, digits) for ele in dict_all[arr0_name]]
        arr1 = [round(ele, digits) for ele in dict_all[arr1_name]]

        score = sum([1 if abs(ele0 - ele1) <= tol else 0 for ele0,
                     ele1 in zip(arr0, arr1)]) / len(arr0)
        score = round(score * 100)

        irr[dict_name][comp_name] = score

    irr = pd.DataFrame(irr).T
    format_table_latex(irr, max1min0=1)
    return irr


def produce_multiple_iccs(dict_all):
    """produce the ICCs for all combinations of rater"""
    icc_res = {}
    combs = get_all_combs(RATERS[:-1])

    for comb in combs:
        loc_name = f'{comb[0]}_{comb[1]}'
        icc_res[loc_name] = intra_class_correlation(
            dict_all, xlsx_name=f'ICC_{comb[0]}_{comb[1]}', compare=comb)

    icc_res_df = pd.DataFrame(icc_res)
    with pd.ExcelWriter(f'../results/analysis_ext/icc.xlsx') as writer:
        icc_res_df.to_excel(writer)

    # first extend the dictionary with the gt
    use_raters = RATERS[:-1]
    iterate = len(dict_all[list(dict_all.keys())[0]])

    for angle in ANGLE_LIST:
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

    icc_res['gt'] = intra_class_correlation(
        dict_all, xlsx_name=f'ICC_{comb[0]}_{comb[1]}', compare=comb)
    icc_res_df = pd.DataFrame(icc_res)

    format_table_latex(icc_res_df, max1min0=1)
    return icc_res


def intra_class_correlation(dict_all, xlsx_name='ICC_all', compare=['net', 'anno', 'excel'], loc_row=4):
    """produce the intra_correlation"""
    icc_res = {}
    with pd.ExcelWriter(f'../results/analysis_ext/{xlsx_name}.xlsx') as writer:
        for cat in ANGLE_LIST:

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

            icc = pg.intraclass_corr(data=data, targets='ID', raters='Rater',
                                     ratings='Scores', nan_policy='omit')
            icc.to_excel(writer, sheet_name=cat)

            val = round(icc.iloc[loc_row]['ICC'], 2)
            ci95 = icc.iloc[loc_row]['CI95%']
            icc_res[cat] = f'{val} [{ci95[0]}, {ci95[1]}]'

    return icc_res


def icc_model_gt(dict_all, use_all=False):
    """define the icc correlation between the ai model and the ground truth"""

    # first extend the dictionary with the gt
    use_raters = RATERS if use_all else RATERS[:-1]
    iterate = len(dict_all[list(dict_all.keys())[0]])

    for angle in ANGLE_LIST:
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

    icc_res_df = pd.DataFrame(icc_res).T
    print(icc_res_df.to_latex())
    return icc_res


def print_extern_infos():
    data = pd.read_excel('../extern_data.xlsx')
    filenames = data['Pat. Nummer']
    side = data['Side']
    demo_data = pd.read_csv('../bga_ext_export_table.csv',
                            sep=';', encoding='unicode_escape')

    pat_infos = {
        'gender': [],
        'age': [],
        'side': [],
    }

    for pat_num, loc_side in zip(filenames, side):
        loc_demo_data = demo_data.loc[demo_data.PIZ == int(pat_num)]
        idx = loc_demo_data.index[0]

        pat_infos['gender'].append(demo_data.iloc[idx]['PatientSex'])
        pat_infos['age'].append(demo_data.iloc[idx]['PatientAge'])
        pat_infos['side'].append(loc_side)

    mean_age, std_age = np.mean(pat_infos['age']), np.std(pat_infos['age'])

    sum_female = np.sum(
        ([1 if loc_gender in 'F' else 0 for loc_gender in pat_infos['gender']]))
    percent_female = (sum_female / len(pat_infos['gender'])) * 100

    sum_left = np.sum(
        ([1 if loc_side in 'Links' else 0 for loc_side in pat_infos['side']]))
    percent_side = (sum_left / len(pat_infos['side'])) * 100

    print('Extern demographic data:')
    print(f'Age: {np.round(mean_age,2)} ± {np.round(std_age,2)}')
    print(f'Female: {sum_female}, {np.round(percent_female,2)} %')
    print(f'Left: {sum_left}, {np.round(percent_side,2)} %')


def osteo_times():
    data = pd.read_excel('../osteotomien.xlsx')
    timekey = 'Zeit Planung '
    data[timekey]

    times = []

    for loc_time in data[timekey]:

        if type(loc_time) != float:
            times.append(loc_time.hour * 60 + loc_time.minute)

    print(
        f'Time Felix: {np.round(np.mean(times), 2)} ± {np.round(np.std(times), 2)} s')


def generate_intra_medicad(mode='felix'):

    df_intra = pd.read_excel('../yannik_intrarater.xlsx')
    if mode == 'felix':
        df_intra = pd.read_excel('../felix_intrarater.xlsx')

    df_medicad = pd.read_excel('../medicad.xlsx')
    pat_ids = [df_intra.iloc[loc_id]['Pat. Nummer']
               for loc_id in range(len(df_intra))]
    medicad_ids = [df_medicad.iloc[loc_id]['Pat. Nummer']
                   for loc_id in range(len(df_medicad))]

    # init the dictionary
    dict_all = {}
    for angle_key in REL_KEY_DICT.keys():
        angle = REL_KEY_DICT[angle_key][0]
        for loc_rater in ['FELIX1', 'FELIX2']:
            dict_all[f'{angle}_{loc_rater}'] = []

    for loc_index, pat_id in enumerate(pat_ids):
        # first find the right loc_df for both dfs
        df_intra_loc = df_intra.iloc[loc_index]
        medicad_index = [index for index, med_id in enumerate(
            medicad_ids) if med_id == pat_id][0]
        df_medicad_loc = df_medicad.iloc[medicad_index]

        for angle_key in REL_KEY_DICT.keys():
            angle_name = REL_KEY_DICT[angle_key][0]

            loc_angle_intra = df_intra_loc[angle_name]
            if type(loc_angle_intra) == str:
                loc_angle_intra = float(loc_angle_intra.replace(
                    ' ', '').replace('°', '').replace(',', '.'))

            loc_angle_medicad = df_medicad_loc[angle_name]
            if type(loc_angle_medicad) == str:
                loc_angle_medicad = float(loc_angle_medicad.replace(
                    ' ', '').replace('°', '').replace(',', '.'))

            dict_all[f'{angle_name}_FELIX1'].append(
                round(loc_angle_intra, 3))
            dict_all[f'{angle_name}_FELIX2'].append(
                round(loc_angle_medicad, 3))

    return dict_all


def generate_intra_slicer():
    with open('../intra.json') as f:
        data_intra = json.load(f)

    with open('../train.json') as f:
        data_train = json.load(f)

    with open('../valid.json') as f:
        data_valid = json.load(f)

    with open('../test.json') as f:
        data_test = json.load(f)

    res_dict = {}
    res_dict2 = {}

    anno_ids = [locdata['image_id'] for locdata in data_intra['annotations']]
    annotations = data_intra['annotations']

    rel_keys = ['bbox', 'keypoints', 'segmentation']

    for image_info in data_intra['images']:
        img_id = image_info['id']
        filename = image_info['file_name']
        loc_anno_indexes = [i for i, x in enumerate(anno_ids) if x == img_id]
        res_dict[filename] = {}

        for loc_anno_index in loc_anno_indexes:
            loc_anno = annotations[loc_anno_index]
            cat_id = loc_anno['category_id']
            cat_name = CATNAMES[cat_id-1]
            loc_keys = list(loc_anno.keys())

            res_dict[filename][cat_name] = {}

            for loc_key in rel_keys:
                if loc_key in loc_keys:
                    res_dict[filename][cat_name][loc_key] = loc_anno[loc_key]

        # now find the filename in train / valid / test
        data2 = {}
        for use_data in [data_train, data_valid, data_test]:
            image_infos2 = use_data['images']
            filenames = [image_info2['file_name']
                         for image_info2 in image_infos2]
            if filename in filenames:
                data2 = use_data

        # now find the image_info with the filename
        image_infos2 = data2['images']
        img_info_id = [i for i, x in enumerate(
            image_infos2) if x['file_name'] == filename][0]
        image_info2 = image_infos2[img_info_id]
        img_id2 = image_info2['id']

        # now the same procedure as above
        anno_ids2 = [locdata2['image_id'] for locdata2 in data2['annotations']]
        annotations2 = data2['annotations']
        loc_anno_indexes2 = [
            i for i, x in enumerate(anno_ids2) if x == img_id2]

        res_dict2[filename] = {}
        for loc_anno_index2 in loc_anno_indexes2:
            loc_anno2 = annotations2[loc_anno_index2]
            cat_id2 = loc_anno2['category_id']
            cat_name2 = CATNAMES[cat_id2-1]
            loc_keys2 = list(loc_anno2.keys())

            res_dict2[filename][cat_name2] = {}

            for loc_key2 in rel_keys:
                if loc_key2 in loc_keys2:
                    res_dict2[filename][cat_name2][loc_key2] = loc_anno2[loc_key2]

    dict_all = {}

    for angle_key in REL_KEY_DICT.keys():
        angle = REL_KEY_DICT[angle_key][0]
        for loc_rater in ['FELIX1', 'FELIX2']:
            dict_all[f'{angle}_{loc_rater}'] = []

    for loc_name in res_dict.keys():
        overall_img = cv2.imread(loc_name)

        anno = res_dict[loc_name]
        anno2 = res_dict2[loc_name]

        anno_res = anno_calculation(
            overall_img, anno, option=2, acc_boost=False, k_len=20)

        anno_res2 = anno_calculation(
            overall_img, anno2, option=2, acc_boost=False, k_len=20)

        for angle_key in REL_KEY_DICT.keys():
            angle_name = REL_KEY_DICT[angle_key][0]
            dict_all[f'{angle_name}_FELIX1'].append(
                round(anno_res[angle_name], 3))
            dict_all[f'{angle_name}_FELIX2'].append(
                round(anno_res2[angle_name], 3))

    # all_angles = pd.DataFrame.from_dict(dict_all)
    return dict_all


def save_dictionary_sorted(dict_all, name):
    d1 = dict(sorted(dict_all.items(), key=lambda x: x[0]))
    all_angles = pd.DataFrame.from_dict(d1)
    all_angles.to_excel(f'../results/{name}.xlsx')


def intrarater_analysis():

    dict_all = generate_intra_medicad(mode='felix')
    icc_res_all1 = produce_multiple_iccs(dict_all)

    dict_all = generate_intra_medicad(mode='yannik')
    icc_res_all2 = produce_multiple_iccs(dict_all)

    intra_all = {}
    intra_all['OS1'] = icc_res_all1['gt']
    intra_all['OS2'] = icc_res_all2['gt'].copy()
    intra_all['AI'] = icc_res_all2['gt']

    for key in intra_all['AI'].keys():
        intra_all['AI'][key] = '1.0 [1.0, 1.0]'

    intra_all_df = pd.DataFrame(intra_all)
    format_table_latex(intra_all_df, max1min0=1)


# %%
if __name__ == '__main__':
    with open('../results/analysis/dict_all_int.json', 'r') as fp:
        dict_all_intern = json.load(fp)

    RATERS = ['OS1', 'OS2', 'OS3', 'AI']
    print_extern_infos()
    preview_external()

    print('EXTERNAL ANALYSIS')
    # build the dictionary containing all data
    dict_all = build_dict_all_from_ext_dataset()

    print('\nAccuracy study GT of OS1-OS3')
    new_acc_study(dict_all, digits=2)

    print('\nBland Altmann Analysis')
    # visualize_all_angles(dict_all)

    print('\nIntraclass Correlation')
    icc_res_all = produce_multiple_iccs(dict_all)

    print('\nClinically relevant Accuracy')
    irr(dict_all, tol=2)

    save_dictionary_sorted(dict_all, 'external_all')

    print('----------------------')
    print('----------------------')
    print('----------------------')
    print('INTERNAL ANALYSIS')

    print('\nAccuracy study GT of OS1-OS3')
    new_acc_study(dict_all_intern, digits=2)

    print('\nBland Altmann Analysis')
    # visualize_all_angles(dict_all_intern, path='../results/analysis')

    print('\nIntraclass Correlation')
    icc_res_all = produce_multiple_iccs(dict_all_intern)

    print('\nClinically relevant Accuracy')
    irr(dict_all_intern, tol=2)

    save_dictionary_sorted(dict_all_intern, 'internall_all')

    print('----------------------')
    print('----------------------')
    print('----------------------')
    print('INTRARATER ANALYSIS')
    RATERS = ['FELIX1', 'FELIX2']
    # intrarater_analysis()


# %%
