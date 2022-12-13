# %%
import json
import numpy as np
"""
Idea: evaluate all jsons:
Get AVERAGE start of object
Get AVERAGE end of object
                in whole picture
Get AVERAGE img size

Then apply average stats on current image - relative or absolute
"""


def get_average_img_dims(cat: str, gen_path='../'):
    """
    iterate trough train dataset and calculate average height and width
    """
    with open(f'{gen_path}jsons/{cat}/train.json') as jf:
        train = json.load(jf)

    images = train['images']
    avg_height = np.array([image['height'] for image in images]).mean()
    avg_width = np.array([image['width'] for image in images]).mean()
    return avg_width, avg_height


def get_average_center(cat: str, gen_path='../'):
    """
    calculate the center of the annotation relative to the cropped image
    """
    with open(f'{gen_path}jsons/{cat}/train.json') as jf:
        train = json.load(jf)

    annos = train['annotations']
    centers_x, centers_y = [], []
    for anno in annos:
        x, y, width, height = anno['bbox']
        center_x = x + 0.5 * width
        center_y = y + 0.5 * height
        centers_x.append(center_x)
        centers_y.append(center_y)

    avg_cx = np.array(centers_x).mean()
    avg_cy = np.array(centers_y).mean()
    return avg_cx, avg_cy


def get_img_cut_dims(det_center: list, cat: str, y_range: list, relative=False, gen_path='../'):
    """return the cut range based on the current detected center"""

    avg_width, avg_height = get_average_img_dims(cat, gen_path=gen_path)
    avg_cx, avg_cy = get_average_center(cat, gen_path=gen_path)

    x_start = det_center[0] - avg_cx
    x_end = x_start + avg_width

    y_start = det_center[1] - avg_cy
    y_end = y_start + avg_height

    # relative
    if relative:
        y_min, y_max = y_range
        cur_height = y_max - y_min
        center_y_per_avg = avg_cy / avg_height

        delta_y = (center_y_per_avg * cur_height) + y_min - det_center[1]

        # test:
        y_start = y_min + delta_y
        y_end = y_max + delta_y

    # ensure the detected center is inside the expected range:
    if (y_range[0] > det_center[1]) or (y_range[1] < det_center[1]):
        # if not keep the expected range
        print(
            f'warning: det. center outside of exp. range for {cat}, keeping exp. range')
        print(f'exp. range: {y_range}')
        print(f'det. center: {det_center[1]}')
        y_start = y_range[0]
        y_end = y_range[1]

    # if distance massively unequal distributed, shorten to middle
    if cat in ['A_up', 'F_t', 'F']:
        y_start, y_end = optimize_img_range_factored(
            det_center, y_start, y_end)

    return [x_start, x_end], [y_start, y_end]


def optimize_img_range_factored(det_center, y_start, y_end, max_factor=3):
    """factorise the distance to top and bottom and decide whether to cut dimensions"""
    dist_top = det_center[1] - y_start
    dist_bottom = y_end - det_center[1]

    # only proceed if det_center within range
    if dist_top > 0 and dist_bottom > 0:

        if dist_top > dist_bottom:

            factor = dist_top / dist_bottom
            if factor > max_factor:

                y_start = det_center[1] - max_factor * dist_bottom

        if dist_top < dist_bottom:

            factor = dist_bottom / dist_top
            if factor > max_factor:

                y_end = det_center[1] + max_factor * dist_top

    return y_start, y_end


# %%
if __name__ == '__main__':
    print(get_average_img_dims('A_low'))
    print(get_average_center('A_low'))
# %%
