# %%
import json
from multiprocessing.sharedctypes import Value
from categories import CATNAMES
from angle_calc import anno_calculation
from PIL import Image, ImageDraw, ImageFont
import cv2
from ipywidgets import widgets
import math
import numpy as np

path_res = '../results/test'

angle_text_dict = {
    "mlpfa": "mLPFA",
    "mldfa": "mLDFA",
    "jlca": "JLCA",
    "mmpta": "mMPTA",
    "mldta": "mLDTA",
    "ama_f": "AMA",
    "mfta": "mFTA",
    "aldfa": "aLDFA",
    "kjlo": "KJLO",
}


def draw_mikulicz(draw: ImageDraw.Draw, angle_dict: dict):
    start = angle_dict['mid_h']
    end = angle_dict['mid_s']
    linedashed(draw, start[0], start[1], end[0], end[1])


def linedashed(draw, x0, y0, x1, y1, dashlen=8, ratio=3):
    """
    https://stackoverflow.com/questions/51908563/dotted-or-dashed-line-with-python-pillow
    """
    dx = x1-x0  # delta x
    dy = y1-y0  # delta y
    # check whether we can avoid sqrt
    if dy == 0:
        length = dx
    elif dx == 0:
        length = dy
    else:
        length = math.sqrt(dx*dx+dy*dy)  # length of line
    xa = dx/length  # x add for 1px line length
    ya = dy/length  # y add for 1px line length
    step = dashlen*ratio  # step to the next dash
    a0 = 0

    while a0 < length:
        a1 = a0+dashlen
        if a1 > length:
            a1 = length
        draw.line((x0+xa*a0, y0+ya*a0, x0+xa*a1, y0+ya*a1),
                  fill=(255, 255, 255, 200), width=6)
        a0 += step


def draw_annos(idx=10, truelabel=False, already_anno=False, angles=True, save=False):
    """"""
    with open(f'{path_res}/test_res.json') as fp:
        data = json.load(fp)

    with open(f'../test.json') as fp:
        data_anno = json.load(fp)

    label = data['label'][idx]
    predict = data['predict'][idx]

    img_anno = cv2.imread(f'{path_res}/{idx}.png')
    img_anno = Image.fromarray(img_anno, 'RGB')

    img = cv2.imread(data_anno['images'][idx]['file_name'])
    img = Image.fromarray(img, 'RGB')

    draw_anno = ImageDraw.Draw(img_anno, 'RGBA')
    draw = ImageDraw.Draw(img, 'RGBA')

    if already_anno:
        img = img_anno
        draw = draw_anno
    if truelabel:
        predict = label

    if angles:
        draw_all_angles(img, draw, predict)
    if save:
        img.save('../current_vis.png')
    return img

def draw_baselines(draw, anno, w=5):
    """draw baselines: femur, tibia, sprunggelenlk"""
    start, end = anno['lat_f'], anno['med_f']
    draw.line((start[0], start[1], end[0], end[1]),
                  fill=(255, 255, 255, 150), width=w)
    start, end = anno['lat_t'], anno['med_t']
    draw.line((start[0], start[1], end[0], end[1]),
                  fill=(255, 255, 255, 150), width=w)
    start, end = anno['lat_s'], anno['med_s']
    draw.line((start[0], start[1], end[0], end[1]),
                  fill=(255, 255, 255, 150), width=w)
    


def draw_all_angles(overall_img, draw, anno, get_jcla=False, acc_boost=True):
    """draw all angles:"""
    angle_dict = anno_calculation(overall_img, anno, acc_boost=acc_boost)
    draw_mikulicz(draw, angle_dict)

    angle_list = ['mfta', 'mlpfa', 'mldfa',
                  'aldfa', 'mmpta', 'mldta',
                  'ama_f', 'jlca', 'kjlo']
    offset_list = [100, 100, 150, 100, 100, 100, 100, 100, 100]
    fill_list = [
        (0, 150, 150, 255),
        (50, 255, 0, 255),
        (50, 50, 255, 255),
        (255, 255, 50, 255),
        (200, 50, 150, 255),
        (200, 255, 0, 255),
        (255, 200, 50, 255),
        (150, 255, 0, 255),
        (225, 100, 50, 255),
    ]
    # first draw ome baselines
    draw_baselines(draw, angle_dict)
    for text_mode in [False, True]:
        for (angle, offset, fill) in zip(angle_list, offset_list, fill_list):
            draw_angle_by_name(draw, angle_dict, angle,
                               fill=fill, offset=offset, text_mode=text_mode)

    draw_miko_load(draw, angle_dict, (237, 62, 69))

    if get_jcla:
        return angle_dict['JLCA']


def draw_miko_load(draw, angle_dict, fill, w=6):
    med_t, lat_t = angle_dict['med_t'], angle_dict['lat_t']
    inter = angle_dict['miko_inter']
    
    try: 
        text = int(round(angle_dict['miko_load']))
    except OverflowError:
        text = 'inf'
    except ValueError:
        text = 'nan'
    
    vis_text = f'Load: {text}%'
    draw.line((med_t[0], med_t[1], lat_t[0], lat_t[1]),
              fill=fill, width=w+2)
    rad = 12
    draw.ellipse((inter[0]-rad, inter[1]-rad,
                  inter[0]+rad, inter[1]+rad), fill=(0, 0, 0, 255))
    rad -= 2
    draw.ellipse((inter[0]-rad, inter[1]-rad,
                  inter[0]+rad, inter[1]+rad), fill=fill)

    tpos = inter.copy()
    tpos[0] -= 150
    tpos[1] += 70
    draw_text(draw, vis_text, tpos, fill, fonts=80)

    fill = (255, 255, 255, 255)

    text = int(angle_dict['Mikulicz'])
    vis_text = f'Mikulicz: {text} mm'
    tpos[0] = 150
    tpos[1] = 100
    draw_text(draw, vis_text, tpos, fill)

    text = int(angle_dict['MAD'])
    vis_text = f'MAD: {text} mm'
    tpos[0] = 150
    tpos[1] = 300
    draw_text(draw, vis_text, tpos, fill)


def draw_angle_by_name(draw, angle_dict, name, fill=(0, 255, 0, 150), offset=100, text_mode=True):
    start = angle_dict[f'{name}_start']
    middle = angle_dict[f'{name}_mid']
    end = angle_dict[f'{name}_end']
    tpos = angle_dict[f'{name}_tpos']
    text = round(angle_dict[name], 1)
    text = f'{name}: {text}°'
    draw_angle(draw, start, middle, end, tpos, text,
               fill=fill, offset=offset, text_mode=text_mode)


def draw_angle(draw, start, middle, end, tpos, text, fill=(0, 255, 0, 150), w=5, offset=100, text_mode=False):

    text_split = text.split(':')
    angle_name = angle_text_dict[text_split[0]]
    vis_text = f'{angle_name}:{text_split[1]}'

    end_a = 180 + math.atan2(middle[1] - start[1],
                             middle[0] - start[0]) * 180 / math.pi
    start_a = math.atan2(end[1] - middle[1], end[0] -
                         middle[0]) * 180 / math.pi

    maxdiff1 = np.sqrt((end[1]-middle[1])**2 + (end[0]-middle[0])**2)
    maxdiff2 = np.sqrt((start[1]-middle[1])**2 + (start[0]-middle[0])**2)

    if end_a - start_a > 360:
        end_a = end_a - 360

    if (end_a - start_a) > 180:
        temp = end_a
        end_a = start_a
        start_a = temp

    angle = end_a - start_a
    if angle < 0 and (angle + 360) > 180:
        temp = end_a
        end_a = start_a
        start_a = temp

    angle = end_a - start_a
    if angle < 0:
        angle += 360

    diff = abs(int(text_split[1].split('.')[0]) / 90)

    offset = min(int(offset / (diff + 1e-5)), 2000,
                 maxdiff1 - 50, maxdiff2 - 50)
    off_arc = offset + 2

    if text_mode:
        draw_text(draw, vis_text, tpos, fill)

    else:
        draw.arc((middle[0] - off_arc, middle[1] - off_arc, middle[0] + off_arc,
                  middle[1] + off_arc), start=start_a, end=end_a, fill=(0, 0, 0, 255), width=3*w + 4)
        draw.line((start[0], start[1], middle[0], middle[1]),
                  fill=(255, 255, 255, 150), width=w)
        draw.line((middle[0], middle[1], end[0], end[1]),
                  fill=(255, 255, 255, 150), width=w)
        draw.arc((middle[0] - offset, middle[1] - offset, middle[0] + offset,
                  middle[1] + offset), start=start_a, end=end_a, fill=fill, width=3*w)


def draw_text(draw, vis_text, tpos, fill,  offi=3, fonts=80):
    shadow_fill = (0, 0, 0, 255)
    try:
        font = ImageFont.truetype("../fonts/ariblk.ttf", fonts)
    except:
        font = ImageFont.load_path("./fonts/ariblk.ttf", fonts)
    
    draw.text((tpos[0]-offi, tpos[1]-offi), vis_text, fill=shadow_fill,
              align='center', font=font)
    draw.text((tpos[0]+offi, tpos[1]-offi), vis_text, fill=shadow_fill,
              align='center', font=font)
    draw.text((tpos[0]-offi, tpos[1]+offi), vis_text, fill=shadow_fill,
              align='center', font=font)
    draw.text((tpos[0]+offi, tpos[1]+offi), vis_text, fill=shadow_fill,
              align='center', font=font)
    draw.text((tpos[0], tpos[1]), vis_text, fill=fill,
              align='center', font=font)


# %%
if __name__ == '__main__':

    idx = widgets.IntSlider(min=0, max=93, value=0)
    widgets.interact(draw_annos, idx=idx)
# %%
