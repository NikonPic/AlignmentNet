# %%
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output
from ipywidgets import widgets
import os
from tqdm import tqdm
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# %%


def read_img(filename: str = '../images_external/10221919.png'):
    """read the img in cv2 format"""
    img = cv2.imread(filename)
    return img


def analyse_img_numbers(img, thresh=80):
    """rotate imgs by 90 degrees, then analyse for valid boxes"""
    # rotate image clockwise first
    d = pytesseract.image_to_data(
        img, output_type=Output.DICT, config='--oem 3 --psm 6', timeout=10)
    n_boxes = len(d['text'])
    valid_boxes = []

    for i in range(n_boxes):
        if int(d['conf'][i].split('.')[0]) > thresh:
            try:
                number = int(d['text'][i])
                (x, y, w, h) = (d['left'][i], d['top']
                                [i], d['width'][i], d['height'][i])
                img = cv2.rectangle(
                    img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                valid_boxes.append({
                    'x': x,
                    'y': y,
                    'num': number,
                })
            except ValueError:
                pass
    return img, valid_boxes


def ocr_analysis(img):
    """analyse the img for numbers and return the factor -> to mm"""
    try:
        fac = perform_analysis(img, vis=False)  # return factor pixel -> cm
        return fac * 10  # pixel -> mm
    except:
        print('error occured on ocr analysis -> no ruler visible')
        return 0.135


def get_grayscale(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.bitwise_not(img)
    return img


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def canny(image):
    return cv2.Canny(image, 100, 200)


def resize(img, factor, factorx=1.5):
    return cv2.resize(img, (0, 0), fx=factor*factorx, fy=factor)


def perform_analysis(img_org, config='--oem 3 --psm 6 outputbase digits', thresh=80, gray=True, base_factor=0.05, factorx=20, vis=False):
    avaiable_nums = {}
    vis_factor = 0.3
    min_list = 4

    img_anno = resize(img_org, vis_factor, 1)
    end_range = factorx

    for main_facor_idx in range(0, 4):
        factor = base_factor + 0.1 * main_facor_idx
        cur_factor = factor / vis_factor

        if len(list(avaiable_nums.keys())) > min_list:
            continue

        for idx in range(0, end_range, 1):
            if len(list(avaiable_nums.keys())) > min_list:
                continue

            loc_factor_x = 0.3 + 0.1 * idx
            img = resize(img_org, factor, loc_factor_x)
            if gray:
                img = get_grayscale(img)
            
            gray = not gray

            d = pytesseract.image_to_data(
                img, output_type=Output.DICT, config=config, timeout=2)
            n_boxes = len(d['text'])

            for i in range(n_boxes):
                loc_text = d['text'][i]

                if len(loc_text) < 2:
                    continue

                if '.' in loc_text or ',' in loc_text or '-' in loc_text or ':' in loc_text:
                    continue

                if int(float(d['conf'][i])) > thresh:
                    (x, y, w, h) = (d['left'][i], d['top']
                                    [i], d['width'][i], d['height'][i])
                    x_put = int(x/loc_factor_x)
                    w_put = int(w/loc_factor_x)

                    # further compress on base-factor:
                    x_put = int(x_put / cur_factor)
                    w_put = int(w_put / cur_factor)
                    y_put = int(y / cur_factor)
                    h_put = int(h / cur_factor)

                    if vis:
                        img_anno = cv2.rectangle(
                            img_anno, (x_put, y_put), (x_put + w_put, y_put + h_put), (255, 0, 0), 2)
                        img_anno = cv2.putText(img_anno, d['text'][i], (x_put, y_put), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(
                            255, 0, 0), thickness=2)

                    if loc_text not in avaiable_nums.keys():
                        avaiable_nums[d['text'][i]] = {
                            'x': x_put,
                            'y': y_put,
                            'w': w_put,
                            'h': h_put,
                            'conf': d['conf'][i],
                        }
                    else:
                        # overwrite if we have a better conf
                        if avaiable_nums[loc_text]['conf'] < d['conf'][i]:
                            avaiable_nums[loc_text]['x'] = x_put
                            avaiable_nums[loc_text]['y'] = y_put
                            avaiable_nums[loc_text]['w'] = w_put
                            avaiable_nums[loc_text]['h'] = h_put
                            avaiable_nums[loc_text]['conf'] = d['conf'][i]

                            if vis:
                                img_anno = cv2.rectangle(
                                    img_anno, (x_put, y_put), (x_put + w_put, y_put + h_put), (0, 255, 0), 2)
                                img_anno = cv2.putText(img_anno, d['text'][i], (x_put, y_put), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(
                                    0, 255, 0), thickness=2)

    img_factor = get_factor(avaiable_nums)
    if vis:
        img_v = Image.fromarray(img_anno)
        return img_factor, img_v

    return img_factor


def get_factor(available_nums, vis_factor=0.3, vis=False, baseline=0.0135):
    """idea compare all possible combinatins of the numbers and return the best one"""
    factors = []

    for num1 in available_nums.keys():
        for num2 in available_nums.keys():
            if num1 == num2:
                continue

            dig1 = int(num1)
            x1 = available_nums[num1]['x'] + 0.5 * available_nums[num1]['w']
            y1 = available_nums[num1]['y'] + 0.5 * available_nums[num1]['h']

            dig2 = int(num2)
            x2 = available_nums[num2]['x'] + 0.5 * available_nums[num2]['w']
            y2 = available_nums[num2]['y'] + 0.5 * available_nums[num2]['h']

            factor = abs(dig1 - dig2) / \
                (np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / vis_factor)
            factors.append(factor)

    factors.sort()
    factors = np.array(factors)
    res = np.median(factors)

    if vis:
        plt.plot(factors)
        plt.show()

    if (abs(res - baseline) / baseline) > 0.4:
        print(f'WARNING: {res} is not close to {baseline}')
        res = baseline

    return res


def do_all_ocrs(ext_files, vis=True):
    factor_list = []
    for idx in tqdm(range(0, len(ext_files))):
        file_ending = ext_files[idx].split('/')[-1]
        fname = ext_files[idx]
        img = read_img(fname)
        if vis:
            loc_factor, img_v = perform_analysis(img, vis=vis)
            img_v.save(f'../ocr_analysis/{file_ending}')
        else:
            loc_factor = perform_analysis(img, vis=vis)
        factor_list.append(loc_factor)

    plt.plot(factor_list)


def do_ocrs(fname):
    """perform ocr on all images"""

    print(fname)
    img = read_img(f'../images_external/{fname}.png')

    loc_factor, img_v = perform_analysis(img, vis=True)
    print(loc_factor)
    return img_v


# %%
if __name__ == '__main__':
    path_ext = '../images_external'
    ext_files = os.listdir(path_ext)
    ext_files = [f'{path_ext}/{f}' for f in ext_files]
    idx = widgets.IntSlider(95, min=0, max=len(ext_files))
    do_all_ocrs(ext_files, vis=True)
    #widgets.interact(do_ocrs, ext_files=ext_files, fname='26903149')


# %%
