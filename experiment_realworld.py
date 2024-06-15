import os
import sys
import time
from util.extract_bbox_from_real_cases import load_bbox_from_txt
from util.evaluate_acc import cal_iou_bbox_list
import logging
from bbh.bbh import BBHNaive, BBHFast
from util.visualization import visualize

import json


def running_time_eva(ann_file_path, out_file_path):
    """
    Evaluate the running time of all the samples in ann_file.
    """
    with open(ann_file_path, 'r') as f:
        img_info = json.load(f)
    for info in img_info:
        bbox_list = img_info[info]['bbox']


def main():
    pass


if __name__ == '__main__':
    main()
