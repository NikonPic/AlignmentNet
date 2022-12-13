# %%


# imports
import numpy as np
import json
import math


def angle_from_vis(start, middle, end):
    end_a = 180 + math.atan2(middle[1] - start[1],
                             middle[0] - start[0]) * 180 / math.pi
    start_a = math.atan2(end[1] - middle[1], end[0] -
                         middle[0]) * 180 / math.pi

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

    return angle


def json_prep_anno(anno: dict, net=True):
    """make the anno dict ready to be json serlialised"""
    json_anno = {}

    # top level
    for key in anno.keys():

        json_anno[key] = {}

        if key == 'imgfac':
            json_anno[key] = anno[key]
            continue

        all_keys = list(set(anno[key].keys()).intersection(
            ['keypoints', 'segmentation', 'bbox']))

        # lower level
        for subkey in all_keys:
            try:
                if isinstance(anno[key][subkey][0], list):
                    json_anno[key][subkey] = []
                    for ele in anno[key][subkey]:
                        if net:
                            json_anno[key][subkey].append([
                                subele.item() for subele in ele])
                        else:
                            json_anno[key][subkey].append([
                                subele for subele in ele])
                else:
                    if net:
                        json_anno[key][subkey] = [ele.item()
                                                  for ele in anno[key][subkey]]
                    else:
                        json_anno[key][subkey] = [ele
                                                  for ele in anno[key][subkey]]
            except:
                print('failed here')

    return json_anno


def get_mid(a, b):
    return [0.5*(a[0] + b[0]), 0.5*(a[1] + b[1])]


def get_per(start, end, per):
    xper = start[0] + per * (end[0] - start[0])
    yper = start[1] + per * (end[1] - start[1])
    return [xper, yper]


def get_difference(a, b):
    return np.linalg.norm(np.array([b[0]-a[0], b[1]-a[1]]))


def signum_fc(x):
    if x < 0:
        return -1
    return 1


class Line():

    def __init__(self, m, b):
        self.m = m
        self.b = b

        self.ortho_m = -1 / self.m

    def intersect(self, line):
        div = (line.m - self.m)
        div = signum_fc(div) * (abs(div) + 1e-7)
        x = (self.b - line.b) / div
        y = self(x)
        return [x, y]

    def __call__(self, x):
        return x * self.m + self.b


class LineP1m(Line):

    def __init__(self, p1: list, m):
        # elevation
        self.m = m

        self.ortho_m = -1 / self.m

        # offset
        self.b = p1[1] - self.m * p1[0]


class LineP1P2(Line):
    """line class to handle all common calculus"""

    def __init__(self, p1: list, p2: list):
        # correct starting order
        if p1[0] < p2[0]:
            self.start_p = p1
            self.end_p = p2
        else:
            self.start_p = p2
            self.end_p = p1

        # elevation
        self.m = (self.end_p[1] - self.start_p[1]) / \
            ((self.end_p[0] - self.start_p[0]) + 1e-6)

        if (abs(self.m) < 1e-6):
            self.m = 1e-6

        self.ortho_m = -1 / self.m

        # offset
        self.b = self.start_p[1] - self.m * self.start_p[0]

        # midpoint of line
        self.midpoint = [0.5 * (self.start_p[0] + self.end_p[0]),
                         0.5 * (self.start_p[1] + self.end_p[1])]

        # further provide a far away points
        self.dist = 4000
        self.dist_min = [self.midpoint[0] - self.dist,
                         self.midpoint[1] - self.dist * self.m]
        self.dist_max = [self.midpoint[0] + self.dist,
                         self.midpoint[1] + self.dist * self.m]


def keyp_by_num(keyp_arr: list, num: int):
    """return the actual keypoint by number selection"""

    lenlist = len(keyp_arr)
    if lenlist <= (num - 1) * 3 + 1:
        return [0, 0]

    x = keyp_arr[(num - 1) * 3]
    y = keyp_arr[(num - 1) * 3 + 1]
    return [x, y]


def improve_key_acc(anno: dict, keyname: str, num: int, scale=0.4, use_max=False):
    """
    Idea: take the keypoint, find the closest matching point in the polygon
    take the scaled mean of both points
    """
    # take the keypoint
    keyp = keyp_by_num(anno[keyname]['keypoints'], num)  # [x, y]
    return improve_key(anno, keyname, keyp, scale, use_max=use_max)


def improve_key(anno: dict, keyname: str, keyp: list, scale=0.4, move_lim=80, use_max=False):
    """
    same as function above, only for the ready keypoint
    scale: 1 -> use keypoint, 0 -> use_segmentation
    """
    keyp = np.array(keyp)
    # get the segmentation
    if 'segmentation' not in anno[keyname].keys():
        return keyp

    segmentation = anno[keyname]['segmentation']
    # now find best match
    match_p = np.array([0, 0])
    min_diff = 1e8

    # iterate trough each polygon
    for poly in segmentation:
        if len(poly) > 2:
            x_arr, y_arr = poly[::2], poly[1::2]
            # iterate trough each point
            for x, y in zip(x_arr, y_arr):
                # get the current difference
                loc_point = np.array([x, y])
                diff = np.linalg.norm(keyp - loc_point)
                if diff < min_diff:
                    min_diff = diff
                    match_p = loc_point

    improved_keyp = scale * keyp + (1-scale) * match_p

    if use_max:
        improved_keyp[1] = max(keyp[1], match_p[1])

    # check the prodeced movement
    diff = get_difference(keyp, match_p)
    if diff > move_lim:
        print(
            f'warning: keypointdiff further than {move_lim}px - no improvement used')
        return keyp

    return improved_keyp.tolist()


def midpoint_calc(medial_point, medial_bottom_point, lateral_bottom_point, lateral_point):
    """
    calculate the midpoint:
    medial_point                          lateral_point
        |                                       |
    medial_bottom_point  ___ midpoint ___ lateral_bottom_point
    """
    # build bottom line
    bottom_line = LineP1P2(medial_bottom_point, lateral_bottom_point)

    # medial line orthogonal to bottom line
    ortho_line_medial = LineP1m(medial_point, bottom_line.ortho_m)
    medial_intersect = bottom_line.intersect(ortho_line_medial)

    # lateral line orthogonal to bottom line
    ortho_line_lateral = LineP1m(lateral_point, bottom_line.ortho_m)
    lateral_intersect = bottom_line.intersect(ortho_line_lateral)

    # final line with middle point
    bottom_line_final = LineP1P2(medial_intersect, lateral_intersect)
    return bottom_line_final.midpoint, medial_intersect, lateral_intersect


def get_hip_midpoint(anno: dict):
    """simpy use H1"""
    keyp = anno['F_t']['keypoints']
    h1 = keyp_by_num(keyp, 1)

    keyp = anno['H']['keypoints']
    h2 = keyp_by_num(keyp, 1)

    diff = get_difference(h1, h2)

    if diff > 30:
        print(f'warning: H1 and H2 differ by {diff}px')
        return h1

    return h2


def get_femur_midpoint(anno: dict, option=1, acc_boost=False):
    """
    option 0: F5
    option 1: F3 - 90° - F1 - MIDPOINT - F2 - 90° - F4
    """
    keyp = anno['F']['keypoints']

    # option 1 -> midpoint
    F1 = keyp_by_num(keyp, 1)  # medial bottom
    F2 = keyp_by_num(keyp, 2)  # lateral bottom
    F3 = keyp_by_num(keyp, 3)  # medial
    F4 = keyp_by_num(keyp, 4)  # lateral

    if acc_boost:
        F1 = improve_key_acc(anno, 'F', 1, use_max=True)
        F2 = improve_key_acc(anno, 'F', 2, use_max=True)
        F3 = improve_key_acc(anno, 'F', 3)
        F4 = improve_key_acc(anno, 'F', 4)

    F1, F2, switch = improve_f1f2(F1, F2, anno)

    if switch:
        F1 = improve_key(anno, 'F', F1)
        F2 = improve_key(anno, 'F', F2)

    # now consider cases:
    norm_f1f2 = get_difference(F1, F2)
    norm_f3f4 = get_difference(F3, F4)

    if norm_f3f4 > norm_f1f2:
        midpoint, medial, lateral = midpoint_calc(F3, F1, F2, F4)
    else:
        midpoint, medial, lateral = midpoint_calc(F1, F1, F2, F2)
        O2 = improve_key_acc(anno, 'O', 2)
        O3 = improve_key_acc(anno, 'O', 3)
        norm_o2o3 = get_difference(O2, O3)

        if norm_o2o3 > norm_f1f2:
            midpoint, medial, lateral = midpoint_calc(O2, F1, F2, O3)

    # option 0 -> F5
    if option == 0:
        midpoint = keyp_by_num(keyp, 5)

   

    return midpoint, medial, lateral


def improve_f1f2(f1, f2, anno, place=0.2):
    """Idea: Improve F1, F2 if both points are on the same condyle"""
    # decide whether the leg is left or right
    left0right1 = get_left0_right1(anno)
    # now define bbox f1f2:
    bbox_femur = anno['F']['bbox']  # [x, y, width, height]

    if len(bbox_femur) < 4:
        print(f'warning: bbox_femur has less than 4 entries, {len(bbox_femur)}')
        return f1, f2, False

    # calculate ccordinates
    f1f2y = bbox_femur[1] + bbox_femur[3]  # bottom line
    fleftx = bbox_femur[0] + place * bbox_femur[2]
    frightx = bbox_femur[0] + (1 - place) * bbox_femur[2]
    x_mid = bbox_femur[0] + 0.5 * bbox_femur[2]

    switch = 0

    # case right F2-F1
    if left0right1:
        f1bbox = [frightx, f1f2y]
        f2bbox = [fleftx, f1f2y]

        # if F1 on wrong side
        if x_mid > f1[0]:
            switch = 1
            f1 = f1bbox

        # if F2 on wrong side:
        if x_mid < f2[0]:
            switch = 1
            f2 = f2bbox

    # case left -> F1-F2
    else:
        f1bbox = [fleftx, f1f2y]
        f2bbox = [frightx, f1f2y]

        # if F1 on wrong side
        if x_mid < f1[0]:
            switch = 1
            f1 = f1bbox

        # if F2 on wrong side:
        if x_mid > f2[0]:
            switch = 1
            f2 = f2bbox

    if switch:
        print('called F1F2 swap')

    return f1, f2, switch


def improve_t1t2(anno: dict, t1, t2, t7, t3, t4, use_mode=True):
    """build t1 / t2 if t1 or t2 failed"""
    # first we need to know if the leg is left or right
    left0right1 = get_left0_right1(anno)
    norm_t1t2 = get_difference(t1, t2)
    norm_t7t1 = get_difference(t7, t1)

    if use_mode:
        key1 = 'T_low'
        key2 = 'T_up'
    else:
        key1 = 'S'
        key2 = 'S'

    # check if the points are on the same side
    if 0.4 * norm_t7t1 > norm_t1t2:
        print('switch activated t1t2')
        # find which point is on the wrong side

        # right leg
        if left0right1:
            if t7[0] < t2[0]:
                print(t2)
                # move T2 "left"
                diffx = 0.5 * abs(t7[0] - t4[0])
                t2[0] = t7[0] - diffx
                print(t2)
                t2 = improve_key(anno, key1, t2, 0)
                print(t2)
            else:
                # move T1 "right"
                diffx = 0.5 * abs(t7[0] - t3[0])
                t1[0] = t7[0] + diffx
                t1 = improve_key(anno, key2, t1, 0)

        else:
            if t7[0] > t2[0]:
                # move T2 "right"
                diffx = 0.5 * abs(t7[0] - t4[0])
                t2[0] = t7[0] + diffx
                t2 = improve_key(anno, key2, t2, 0)
            else:
                # move T1 "left"
                diffx = 0.5 * abs(t7[0] - t3[0])
                t1[0] = t7[0] - diffx
                t1 = improve_key(anno, key2, t1, 0)

    return t1, t2


def improve_t3t4t5t6(anno: dict, t3, t4, t5, t6, t7):
    """catch error if t3 and t5 are both on the wrong side"""
    left0right1 = get_left0_right1(anno)
    norm_t3t7 = get_difference(t3, t7)
    norm_t3t4 = get_difference(t3, t4)
    norm_t5t6 = get_difference(t5, t6)

    # if all points have failed
    if 0.4 * norm_t3t7 > max(norm_t3t4, norm_t5t6):
        print('activated t3456 switch')
        # find which point is on the wrong side
        diffx = 2 * abs(t7[0] - t3[0])
        print(diffx)

        # right leg
        if left0right1:

            # t4 and 6 wrong
            if t4[0] > t7[0]:
                print('t4 and 6 wrong, right leg')
                t4[0] = t7[0] - diffx
                t4 = improve_key(anno, 'T_up', t4, 0)

                t6[0] = t7[0] - diffx
                t6 = improve_key(anno, 'T_up', t6, 0)

            # t3 and 5 wrong
            else:
                t3[0] = t7[0] + diffx
                t3 = improve_key(anno, 'T_up', t3, 0)

                t5[0] = t7[0] + diffx
                t5 = improve_key(anno, 'T_up', t5, 0)

        # left leg
        else:

            # t4 and 6 wrong
            if t4[0] < t7[0]:
                t4[0] = t7[0] + diffx
                t4 = improve_key(anno, 'T_up', t4, 0)

                t6[0] = t7[0] + diffx
                t6 = improve_key(anno, 'T_up', t6, 0)

            # t3 and 5 wrong
            else:
                t3[0] = t7[0] - diffx
                t3 = improve_key(anno, 'T_up', t3, 0)

                t5[0] = t7[0] - diffx
                t5 = improve_key(anno, 'T_up', t5, 0)

    return t3, t4, t5, t6


def get_tibia_midpoint(anno: dict, option=1, acc_boost=False):
    """
    option 0: T7
    option 1: T5 - 90° - T1 - MIDPOINT - T2 - 90° - T6
    option 2: T5 - 90° - T3 - MIDPOINT - T4 - 90° - T6
    """
    keyp = anno['T_low']['keypoints']

    # option 1 -> midpoint
    T1 = keyp_by_num(keyp, 1)  # medial bottom 1
    T2 = keyp_by_num(keyp, 2)  # lateral bottom 1
    T3 = keyp_by_num(keyp, 3)  # medial bottom 2
    T4 = keyp_by_num(keyp, 4)  # lateral bottom 2
    T5 = keyp_by_num(keyp, 5)  # medial
    T6 = keyp_by_num(keyp, 6)  # lateral

    keyp_up = anno['T_up']['keypoints']
    T7 = keyp_by_num(keyp_up, 1)  # medial bottom 1

    if acc_boost:
        T1 = improve_key(anno, 'T_up', T1)
        T2 = improve_key(anno, 'T_up', T2)
        T3 = improve_key(anno, 'T_up', T3)
        T4 = improve_key(anno, 'T_up', T4)
        T5 = improve_key(anno, 'T_up', T5)
        T6 = improve_key(anno, 'T_up', T6)

    T3, T4, T5, T6 = improve_t3t4t5t6(anno, T3, T4, T5, T6, T7)
    T1, T2 = improve_t1t2(anno, T1, T2, T7, T3, T4)

    # finally use option 2
    midpoint, medial, lateral = midpoint_calc(T5, T3, T4, T6)

    norm_t1t2 = get_difference(T1, T2)
    norm_t3t4 = get_difference(T3, T4)
    norm_t5t6 = get_difference(T5, T6)

    # t3t4 together
    if norm_t3t4 < 0.4 * norm_t1t2:
        print('t5-t1-t2-t6')
        midpoint, medial, lateral = midpoint_calc(T5, T1, T2, T6)
    # t5t6 together
    if norm_t5t6 < 0.4 * norm_t3t4:
        print('t3-t1-t2-t4')
        T4 = improve_lateral_tibia(anno, T4)
        midpoint, medial, lateral = midpoint_calc(T3, T1, T2, T4)
    # t3t4 and t5t6 together
    if norm_t5t6 < norm_t1t2 and norm_t3t4 < norm_t1t2:
        print('t1-t1-t2-t2')
        midpoint, medial, lateral = midpoint_calc(T1, T1, T2, T2)

    # option 0 -> F5
    if option == 0:
        midpoint = keyp_by_num(anno['T_up']['keypoints'], 1)

    if option == 1:
        midpoint, medial, lateral = midpoint_calc(T5, T1, T2, T6)

    return midpoint, medial, lateral


def improve_lateral_tibia(anno, lateral):
     # further get Fi3
    keyp_fi = anno['Fi_up']['keypoints']
    fi_3 = keyp_by_num(keyp_fi, 1)
    l0r1 = get_left0_right1(anno)

    # case right
    if l0r1 == 1:
        lateral[0] = max(lateral[0], fi_3[0])
    else:
        lateral[0] = min(lateral[0], fi_3[0])
    return lateral

def get_sprung_midpoint(anno: dict, option=1, acc_boost=False):
    """
    option 0: S5
    option 1: S3 - 90° - S1 - MIDPOINT - S2 - 90° - S4
    """
    keyp = anno['S']['keypoints']

    # option 0 -> F5
    if option == 0:
        return keyp_by_num(keyp, 5)

    # option 1 -> midpoint
    S1 = keyp_by_num(keyp, 1)  # medial bottom
    S2 = keyp_by_num(keyp, 2)  # lateral bottom
    S3 = keyp_by_num(keyp, 3)  # medial
    S4 = keyp_by_num(keyp, 4)  # lateral
    S5 = keyp_by_num(keyp, 5)  # center

    if acc_boost:
        S1 = improve_key_acc(anno, 'S', 1)
        S2 = improve_key_acc(anno, 'S', 2)
        S3 = improve_key_acc(anno, 'S', 3)
        S4 = improve_key_acc(anno, 'S', 4)
        S5 = improve_key_acc(anno, 'S', 5)

    norm_s1s2 = get_difference(S1, S2)
    norm_s3s4 = get_difference(S3, S4)
    norm_s3s5 = get_difference(S3, S5)

    # detect switches
    S1, S2 = improve_t1t2(anno, S1, S2, S5, S3, S4, use_mode=False)

    if norm_s3s4 > norm_s1s2:
        midpoint, medial, lateral = midpoint_calc(S3, S1, S2, S4)

        if norm_s3s5 > norm_s1s2:
            midpoint = S5

    else:
        midpoint, medial, lateral = midpoint_calc(S1, S1, S2, S2)

    return midpoint, medial, lateral


def get_anatomical_axis(anno: dict, key='A_up'):
    """
    A1-mid-A2
    A3-mid-A4
    """
    # decide whether the leg is left or right
    left0right1 = get_left0_right1(anno)

    bbox = anno[key]['bbox']
    if len(bbox) < 4:
        bbox = [0, 0, 0, 0]

    keyp = anno[key]['keypoints']
    A1, A2, A3, A4 = keyp_by_num(keyp, 1), keyp_by_num(
        keyp, 2), keyp_by_num(keyp, 3), keyp_by_num(keyp, 4)

    norm12 = get_difference(A1, A2)
    norm34 = get_difference(A3, A4)
    norm13 = get_difference(A1, A3)
    norm24 = get_difference(A2, A4)

    diffy12 = abs(A1[1] - A2[1])
    diffy34 = abs(A3[1] - A4[1])
    diffy13 = abs(A1[1] - A3[1])

    # case A1 or A3 failed
    if (diffy13 < diffy12) and (norm13 < norm12):
        # A3 failed
        if diffy12 < diffy34:
            A3 = [A1[0], A4[1]]
        # A1 failed
        else:
            A1 = [A3[0], A2[1]]

    diffy12 = abs(A1[1] - A2[1])
    diffy34 = abs(A3[1] - A4[1])
    diffy24 = abs(A2[1] - A4[1])

    # case A2 or A4 failed
    if (diffy24 < diffy34) and (norm24 < norm34):
        # A4 failed
        if diffy12 < diffy34:
            A4 = [A2[0], A3[1]]
        # A2 failed
        else:
            A2 = [A4[0], A1[1]]

    # case right
    if left0right1:
        diffa1 = abs(A1[0] - (bbox[0] + bbox[2]))
        diffa2 = abs(A2[0] - bbox[0])
        diffa3 = abs(A3[0] - (bbox[0] + bbox[2]))
        diffa4 = abs(A4[0] - bbox[0])

    # case left
    else:
        diffa1 = abs(A1[0] - bbox[0])
        diffa2 = abs(A2[0] - (bbox[0] + bbox[2]))
        diffa3 = abs(A3[0] - bbox[0])
        diffa4 = abs(A4[0] - (bbox[0] + bbox[2]))

    # case A1 or A2 failed
    if (norm12 < 0.14 * norm13):
        midx = (A1[0] + A2[0]) / 2
        midsum = abs(A1[0] - midx) + abs(A2[0] - midx)

        if midsum < 0.14 * bbox[2]:
            A2 = [bbox[0] + (1-left0right1) * bbox[2], A2[1]]

        # case A2 failed
        elif diffa1 < diffa2:
            A2 = [bbox[0] + (1 - left0right1) * bbox[2], A1[1]]
        # case A1 failed
        else:
            A1 = [bbox[0] + left0right1 * bbox[2], A2[1]]

    # case A3 or A4 failed
    if (norm34 < 0.14 * norm13):
        # case A4 failed
        if diffa3 < diffa4:
            A4 = [bbox[0] + (1 - left0right1) * bbox[2], A3[1]]
        # case A1 failed
        else:
            A3 = [bbox[0] + left0right1 * bbox[2], A4[1]]

    # further check x compatibility
    diffx12 = abs(A1[0] - A2[0])
    diffx13 = abs(A1[0] - A3[0])
    diffx34 = abs(A3[0] - A4[0])
    diffx24 = abs(A2[0] - A4[0])

    # case A1 or A2 failed
    if diffx12 < 0.14 * diffx34:
        # case A2 failed
        if diffx13 < diffx24:
            A2 = [A4[0], A2[1]]
        # case A1 failed
        else:
            A1 = [A3[0], A1[1]]

    # case A3 or A4 failed
    if diffx34 < 0.14 * diffx12:
        # case A3 failed
        if diffx24 < diffx13:
            A3 = [A1[0], A3[1]]
        # case A4 failed
        else:
            A4 = [A2[0], A4[1]]

    mid1 = get_mid(A1, A2)
    mid2 = get_mid(A3, A4)
    return [mid1, mid2]


def get_mechanical_axis_femur(anno_results: dict):
    """
    mid_h - (intersect mid_f with F1-F2)
    """
    med_f, lat_f = anno_results['med_f'], anno_results['lat_f']
    mid_h = anno_results['mid_h']
    mid_f = anno_results['mid_f']

    line1 = LineP1P2(mid_h, mid_f)
    line2 = LineP1P2(med_f, lat_f)
    new_mid_f = line1.intersect(line2)

    return new_mid_f


def get_mechanical_axis_tibia(anno_results: dict):
    """
    mid_s - (intersect mid_t with T1-T2)
    """
    med_t, lat_t = anno_results['med_t'], anno_results['lat_t']
    mid_s = anno_results['mid_s']
    mid_t = anno_results['mid_t']

    line1 = LineP1P2(mid_s, mid_t)
    line2 = LineP1P2(med_t, lat_t)
    new_mid_t = line1.intersect(line2)

    return new_mid_t


def alternative(a, b, c):
    """
    https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    """
    # to numpy
    a, b, c = np.array(a), np.array(b), np.array(c)

    # the relevant vecors
    ba, bc = a - b, c-b

    # now the main calculus
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def angle_between_points(a, b, c):
    return (math.atan2(c[1]-b[1], c[0] - b[0]) - math.atan2(b[1]-a[1], b[0] - a[0])) * (180 / math.pi)


def get_left0_right1(anno):
    """determine whether the leg is left(0) or right(1)"""
    mid_h = get_hip_midpoint(anno)

    # [x(left), y(top), width, height]
    bbox = anno['F_t']['bbox']

    if len(bbox) < 4:
        bbox = [0, 0, 0, 0]

    # determine whether hip is left or right:
    bbox_mid_x = bbox[0] + 0.5 * bbox[2]

    # case ft right:
    if bbox_mid_x > mid_h[0]:
        return 0  # left
    # case ft left
    else:
        return 1  # right


def improve_f_t(anno, anno_results):
    """if the difference between mid_h and f_t is too small -> use """
    mid_h = anno_results['mid_h']
    ft = keyp_by_num(anno['F_t']['keypoints'], 2)

    # [x(left), y(top), width, height]
    bbox = anno['F_t']['bbox']
    if len(bbox) < 4:
        bbox = [0, 0, 0, 0]

    # determine whether hip is left or right:
    bbox_mid_x = bbox[0] + 0.5 * bbox[2]

    # case ft right:
    if bbox_mid_x > mid_h[0]:
        ftbbox = [bbox[0] + bbox[2], ft[1]]
    # case ft left
    else:
        ftbbox = [bbox[0], ft[1]]

    # distance between hip and the detected ft point
    dist_h_ft = get_difference(mid_h, ft)
    dist_ft_ffbbox = get_difference(ft, ftbbox)

    # if dist between hip and ft is smaller than the two fts
    if dist_h_ft < dist_ft_ffbbox:
        print('called ft improvement')
        ft = ftbbox
        if 'F_t2' in anno.keys():
            print('using the other side as reference')
            ft2 = keyp_by_num(anno['F_t2']['keypoints'], 2)
            ft = [ftbbox[0], ft2[1]]

    return ft


def anno_calculation(overall_img, anno: dict, option=2, acc_boost=False, k_len=25):
    """perform all angle calculations"""
    anno_results = {}
    anno_results['img_shape'] = overall_img.shape
    anno_results['l0r1'] = get_left0_right1(anno)
    anno_results['lm1r1'] = 1 if anno_results['l0r1'] else -1

    # get the midpoints of each joint
    anno_results['mid_h'] = get_hip_midpoint(anno)
    anno_results['mid_f'], anno_results['med_f'], anno_results['lat_f'] = get_femur_midpoint(
        anno, option=option, acc_boost=acc_boost)
    anno_results['mid_t'], anno_results['med_t'], anno_results['lat_t'] = get_tibia_midpoint(
        anno, option=option, acc_boost=acc_boost)
    anno_results['mid_s'], anno_results['med_s'], anno_results['lat_s'] = get_sprung_midpoint(
        anno, option=option, acc_boost=acc_boost)

    # some relevant points:
    anno_results['f_t'] = improve_f_t(anno, anno_results)

    # get the axis
    anno_results['ana_f'] = get_anatomical_axis(anno, key='A_up')
    anno_results['ana_t'] = get_anatomical_axis(anno, key='A_low')
    anno_results['mid_f'] = get_mechanical_axis_femur(anno_results)
    anno_results['mid_t'] = get_mechanical_axis_tibia(anno_results)

    # combined angles
    anno_results['MAD'] = mad(anno_results, anno, k_len)  # yes
    anno_results['Mikulicz auf TP'] = load_tibia(anno_results, anno)  # yes

    # calculate the femur
    anno_results['mLPFA'] = mlpfa(anno_results, anno)  # yes
    anno_results['AMA'] = ama_f(anno_results, anno)  # yes
    anno_results['JLCA'] = jlca(anno_results)  # yes
    anno_results['aLDFA'] = aldfa(anno_results)  # yes
    anno_results['mLDFA'] = mldfa(anno_results)  # yes

    # tibia angles
    anno_results['mMPTA'] = mmpta(anno_results)  # yes
    anno_results['mLDTA'] = mldta(anno_results)  # yes
    anno_results['KJLO'] = kjlo(anno_results)  # yes

    # central angle
    anno_results['mFTA'] = mech_tibiofemoral(anno_results)  # yes

    return anno_results


def kjlo(anno_results: dict):
    """
    angle: line(med_t - lat_t) - line(horizontal)
    """
    # get the tibial line
    med_t = anno_results['med_t']
    lat_t = anno_results['lat_t']
    l_tibia = LineP1P2(med_t, lat_t)

    if med_t[1] > lat_t[1]:
        p1 = med_t
        p2 = lat_t
    else:
        p1 = lat_t
        p2 = med_t

    # get the horizontal line
    hor_1 = [p1[0], p1[1]]
    hor_2 = [p2[0], p1[1]]
    hor_line = LineP1P2(hor_1, hor_2)

    # get the interception
    intersect = l_tibia.intersect(hor_line)

    # get the angle
    kjlo = angle_from_vis(p2, intersect, hor_2)
    anno_results['kjlo_start'] = p2
    anno_results['kjlo_mid'] = intersect
    anno_results['kjlo_end'] = hor_2
    anno_results['kjlo_tpos'] = hor_2.copy()
    anno_results['kjlo_tpos'][1] += 300
    anno_results['kjlo'] = kjlo
    ensure_visibility(anno_results, 'kjlo_tpos')

    return kjlo


def mlpfa(anno_results: dict, anno: dict):
    """
    angle: mid_f -> mid_h -> F6
    """
    mid_f = anno_results['mid_f']
    mid_h = anno_results['mid_h']
    f_t = anno_results['f_t']

    mlpfa = angle_from_vis(mid_f, mid_h, f_t)

    anno_results['mlpfa_start'] = mid_f
    anno_results['mlpfa_mid'] = mid_h
    anno_results['mlpfa_end'] = f_t
    anno_results['mlpfa_tpos'] = mid_h.copy()
    anno_results['mlpfa_tpos'][1] += 300
    anno_results['mlpfa'] = mlpfa
    ensure_visibility(anno_results, 'mlpfa_tpos')

    return mlpfa


def mldfa(anno_results: dict):
    """
    angle: lat_f -> mid_f -> mid_h
    """
    mid_h = anno_results['mid_h']
    lat_f = anno_results['lat_f']
    mid_f = anno_results['mid_f']

    mldfa = angle_from_vis(lat_f, mid_f, mid_h)

    anno_results['mldfa_start'] = lat_f
    anno_results['mldfa_mid'] = mid_f
    anno_results['mldfa_end'] = mid_h
    anno_results['mldfa_tpos'] = mid_f.copy()
    anno_results['mldfa_tpos'][1] -= 600
    anno_results['mldfa_tpos'] = optimize_pos('mldfa_tpos', anno_results)
    anno_results['mldfa'] = mldfa
    ensure_visibility(anno_results, 'mldfa_tpos')

    return mldfa


def ama_f(anno_results: dict, anno: dict, use_f5=False):
    """
    anatomic mechanical angle femur
    h1 ->  intersection -> Atop
    """
    mid_h = anno_results['mid_h']
    mid_f = anno_results['mid_f']

    # alternative, not quite succesfull
    if use_f5:
        keyp = anno['F']['keypoints']
        f5 = keyp_by_num(keyp, 5)
        mid_f = f5

    line1 = LineP1P2(mid_h, mid_f)

    top_a = anno_results['ana_f'][0]
    bottom_a = anno_results['ana_f'][1]
    line2 = LineP1P2(top_a, bottom_a)

    inter = line1.intersect(line2)

    ama_f = abs(limit_angle(angle_between_points(mid_h, inter, top_a)))

    anno_results['ama_f_start'] = mid_h
    anno_results['ama_f_mid'] = inter
    anno_results['ama_f_end'] = top_a
    anno_results['ama_f_tpos'] = top_a.copy()
    anno_results['ama_f_tpos'][0] += 100
    anno_results['ama_f'] = ama_f
    ensure_visibility(anno_results, 'ama_f_tpos')

    return ama_f


def aldfa(anno_results: dict):
    """
    anatomic mechanical angle femur
    top_a ->  mid_f -> lat_f
    """
    top_a = anno_results['ana_f'][0]
    bottom_a = anno_results['ana_f'][1]
    line1 = LineP1P2(top_a, bottom_a)

    med_f = anno_results['med_f']
    lat_f = anno_results['lat_f']
    line2 = LineP1P2(med_f, lat_f)

    inter = line1.intersect(line2)

    aldfa = abs(limit_angle(angle_between_points(top_a, inter, med_f)))

    anno_results['aldfa_start'] = top_a
    anno_results['aldfa_mid'] = inter
    anno_results['aldfa_end'] = lat_f
    anno_results['aldfa_tpos'] = lat_f.copy()
    anno_results['aldfa_tpos'][1] -= 800
    anno_results['aldfa'] = aldfa
    ensure_visibility(anno_results, 'aldfa_tpos')

    return aldfa


def limit_angle(angle):
    """
    fit angle between -180 and 180 degrees
    """
    # case: 270 -> -90
    if angle > 180:
        angle = angle - 360

    # case: -270 -> 90
    if angle < -180:
        angle = angle + 360

    if angle < -90:
        angle = angle + 180

    if angle > 90:
        angle = angle - 180

    return angle


def jlca(anno_results: dict):
    """
    angle: F1-F2 <-> (T3-T4  or T1-T2)
    """
    lm1r1 = anno_results['lm1r1']
    med_f, lat_f = anno_results['med_f'], anno_results['lat_f']
    med_t, lat_t = anno_results['med_t'], anno_results['lat_t']
    mid_f, mid_t = anno_results['mid_f'], anno_results['mid_t']

    left_0_right_1 = 0
    if med_f[0] > lat_f[0]:
        left_0_right_1 = 1

    line1 = LineP1P2(med_f, lat_f)
    line2 = LineP1P2(med_t, lat_t)
    inter = line1.intersect(line2)

    sign = -1
    # right
    if left_0_right_1:
        if inter[0] > mid_t[0]:
            sign = 1
    # left
    else:
        if inter[0] < mid_t[0]:
            sign = 1

    jlca = sign * angle_from_vis(lat_f, inter, lat_t)

    # catch too small jlca
    if jlca < -150:
        jlca += 180
        jlca *= -1
    
    # include abs
    jlca = abs(jlca)

    anno_results['jlca_start'] = mid_f
    anno_results['jlca_mid'] = inter
    anno_results['jlca_end'] = mid_t
    anno_results['jlca_tpos'] = lat_f.copy()
    anno_results['jlca_tpos'][0] += lm1r1 * 200
    anno_results['jlca_tpos'][1] -= 450
    anno_results['jlca'] = jlca
    ensure_visibility(anno_results, 'jlca_tpos')

    return jlca


def mmpta(anno_results: dict):
    """
    angle: mid_s -> mid_t -> med_t
    """
    mid_s, mid_t, lat_t = anno_results['mid_s'], anno_results['mid_t'], anno_results['lat_t']
    med_t = anno_results['med_t']
    mmpta = angle_from_vis(mid_s, mid_t, med_t)
    lm1r1 = anno_results['lm1r1']

    anno_results['mmpta_start'] = mid_s
    anno_results['mmpta_mid'] = mid_t
    anno_results['mmpta_end'] = med_t
    anno_results['mmpta_tpos'] = med_t.copy()
    anno_results['mmpta_tpos'][1] += 500
    anno_results['mmpta_tpos'][0] -= lm1r1 * 250
    anno_results['mmpta'] = mmpta
    ensure_visibility(anno_results, 'mmpta_tpos')

    return mmpta


def mldta(anno_results: dict):
    """
    angle: mid_t -> mid_s -> lat_s
    """
    mid_t, mid_s, lat_s = anno_results['mid_t'], anno_results['mid_s'], anno_results['lat_s']
    mldta = angle_from_vis(mid_t, mid_s, lat_s)

    anno_results['mldta_start'] = mid_t
    anno_results['mldta_mid'] = mid_s
    anno_results['mldta_end'] = lat_s
    anno_results['mldta_tpos'] = lat_s.copy()
    anno_results['mldta_tpos'][0] -= 200
    anno_results['mldta_tpos'][1] -= 300
    anno_results['mldta'] = mldta
    ensure_visibility(anno_results, 'mldta_tpos')

    return mldta


def mad(anno_results: dict, anno: dict, k_len):
    """
    mikulicz line and the difference there
    """
    # first get the disctance for 25mm:
    keyp = anno['K']['keypoints']
    if len(keyp) > 5:
        K1, K2 = keyp_by_num(keyp, 1), keyp_by_num(keyp, 2)
        imgfac = k_len / get_difference(K1, K2)
        # use bounding box instead!
        if imgfac > 1e6:
            _, _, _, height_k = anno['K']['bbox']
            imgfac = k_len / height_k
    else:
        imgfac = 0

    if 'imgfac' in anno.keys():
        imgfac2 = anno['imgfac']
        if math.isnan(imgfac2):
            imgfac2 = 0
    else:
        imgfac2 = 0

    baseline = 0.14
    imgfac_ruler = imgfac2
    imgfac_sphere = imgfac
    facdiff1 = abs(imgfac - baseline) / baseline
    facdiff2 = abs(imgfac2 - baseline) / baseline

    if facdiff2 < facdiff1:
        imgfac = imgfac2

    if min(facdiff1, facdiff2) > 0.5:
        print('mad: warning: imgfac difference is too big, using baseline')
        print(imgfac_ruler)
        print(imgfac_sphere)
        print(imgfac)
        imgfac = baseline

    anno_results['imgfac'] = imgfac
    mid_h, mid_s, mid_t = anno_results['mid_h'], anno_results['mid_s'], anno_results['mid_t']
    mikulicz = LineP1P2(mid_h, mid_s)
    ortho_line_mikolicz = LineP1m(mid_t, mikulicz.ortho_m)
    intersect = mikulicz.intersect(ortho_line_mikolicz)
    miko_len = get_difference(mid_h, mid_s) * imgfac
    anno_results['Mikulicz'] = miko_len
    offset = get_difference(intersect, mid_t)

    # decide on signum
    med_t, lat_t = anno_results['med_t'], anno_results['lat_t']
    signum = 1
    diff1 = get_difference(intersect, med_t)
    diff2 = get_difference(intersect, lat_t)
    if diff2 < diff1:
        signum = -1

    return offset * imgfac * signum


def load_tibia(anno_results: dict, anno: dict):
    """
    percentage of load on tibia from 0% med to 100% lateral
    """
    left0_right1 = get_left0_right1(anno)

    mid_h, mid_s = anno_results['mid_h'], anno_results['mid_s']
    mikulicz = LineP1P2(mid_h, mid_s)

    med_t, lat_t = anno_results['med_t'], anno_results['lat_t']
    tibialine = LineP1P2(med_t, lat_t)

    intersect = mikulicz.intersect(tibialine)

    sign = 1

    if left0_right1 == 0 and intersect[0] < med_t[0]:
        sign = -1

    if left0_right1 == 1 and intersect[0] > med_t[0]:
        sign = -1

    load = sign * (get_difference(med_t, intersect) /
                   get_difference(med_t, lat_t)) * 100

    anno_results['miko_load'] = load
    anno_results['miko_inter'] = intersect
    anno_results['miko_inter_tpos'] = intersect.copy()
    ensure_visibility(anno_results, 'miko_inter_tpos')

    if load > 100 or load < 0:
        load = 0

    return load


def mech_tibiofemoral(anno_results: dict):
    """
    angle between both mechanical axis
    """
    mid_h, mid_f = anno_results['mid_h'], anno_results['mid_f']
    mid_t, mid_s = anno_results['mid_t'], anno_results['mid_s']

    fline = LineP1P2(mid_h, mid_f)
    tline = LineP1P2(mid_t, mid_s)

    intersect = fline.intersect(tline)
    load = anno_results['miko_load']

    signum = -1
    if load > 50:
        signum = 1

    mfta = signum * \
        abs(limit_angle(angle_between_points(mid_s, intersect, mid_h)))

    if intersect[1] > mid_t[1]:
        endp = mid_h.copy()
        startp = mid_t.copy()
    else:
        endp = mid_s.copy()
        startp = mid_f.copy()

    anno_results['mfta_start'] = startp
    anno_results['mfta_mid'] = intersect
    anno_results['mfta_end'] = endp
    anno_results['mfta_tpos'] = intersect.copy()
    anno_results['mfta_tpos'][0] -= 100
    anno_results['mfta_tpos'][1] += 200
    anno_results['mfta_tpos'] = optimize_pos('mfta_tpos', anno_results)
    anno_results['mfta'] = mfta
    ensure_visibility(anno_results, 'mfta_tpos')

    return mfta


def ensure_visibility(anno_results: dict, key: str, min_pos=200, max_pos=1000):
    """
    ensure that the keypoint is visible in the image
    """
    width = anno_results['img_shape'][1]
    pos = anno_results[key][0]

    if pos < min_pos:
        anno_results[key][0] = min_pos

    if (pos > (width - max_pos)):
        anno_results[key][0] = width - max_pos


def optimize_pos(key, anno_results: dict):
    """
    go trough all tpos and define if the current pos needs adaption
    """
    mypos = anno_results[key].copy()
    all_keys = anno_results.keys()
    pos_keys = [loc_key for loc_key in all_keys if '_tpos' in loc_key]
    pos_keys.remove(key)

    for diff in range(120, 0, -1):
        min_abs_diff = 1e8
        rel_pos = [0, 0]

        # get the minimal difference
        for key in pos_keys:
            other_pos = anno_results[key]
            if abs(mypos[1] - other_pos[1]) < min_abs_diff:
                min_abs_diff = abs(mypos[1] - other_pos[1])
                rel_pos = other_pos.copy()

        # move the current pos away from mindiff
        if min_abs_diff < 200:

            if rel_pos[1] > mypos[1]:
                mypos[1] -= diff
            else:
                mypos[1] += diff

    return mypos


# %%
if __name__ == '__main__':
    with open('../example_anno.json', 'r') as fp:
        anno = json.load(fp)

    anno_results = anno_calculation(None, anno)
    print(anno_results)
# %%
