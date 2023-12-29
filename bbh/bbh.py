import numpy as np


def uncovered_area_metric(bbox_s, bbox_t):
    x_tl, y_tl = min(bbox_s[0], bbox_t[0]), min(bbox_s[1], bbox_t[1])
    x_br, y_br = max(bbox_s[2], bbox_t[2]), max(bbox_s[3], bbox_t[3])
    area_s = (bbox_s[2]-bbox_s[0])*(bbox_s[3]-bbox_s[1])
    area_t = (bbox_t[2]-bbox_t[0])*(bbox_t[3]-bbox_t[1])
    area_st = (x_br-x_tl)*(y_br-y_tl)
    return area_st - area_s - area_t


class BBH:
    def __init__(self,
                 bboxes,
                 dist_metric=uncovered_area_metric):
        pass


class BBHNaive(BBH):
    def __init__(self):
        pass


class BBHFast(BBH):
    def __init__(self):
        pass