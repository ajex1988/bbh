import os
import sys
import time
from util.extract_bbox_from_real_cases import load_bbox_from_txt
from util.evaluate_acc import cal_iou_bbox_list
import logging
from bbh.bbh import BBHNaive, BBHFast
from util.visualization import visualize

import json


def running_time_eva(ann_file_path):
    """
    Evaluate the running time of all the samples in ann_file.
    """
    with open(ann_file_path, 'r') as f:
        img_info = json.load(f)
    bf_perf = {}
    fast_perf = {}
    for info in img_info:
        bbox_list = img_info[info]['bbox_list']
        alg_bf = BBHNaive(bboxes=bbox_list)
        alg_fast = BBHFast(bboxes=bbox_list)

        t_start = time.perf_counter()
        _ = alg_bf.merge()
        t_end = time.perf_counter()
        bf_perf[info] = {"time": t_end - t_start,
                         "count": len(img_info[info]['bbox_list'])}

        t_start = time.perf_counter()
        _ = alg_fast.merge()
        t_end = time.perf_counter()
        fast_perf[info] = {"time": t_end - t_start,
                           "count": len(img_info[info]['bbox_list'])}
    t_average = 0
    cnt = 0
    for info in bf_perf:
        t_average += bf_perf[info]["time"]
        cnt += bf_perf[info]["count"]
    t_average = t_average / len(bf_perf)
    cnt = cnt / len(bf_perf)
    print("Brute Force Evaluation")
    print(f"Average running time: {t_average}")
    print(f"Average bbox: {cnt}")

    t_average = 0
    cnt = 0
    for info in fast_perf:
        t_average += fast_perf[info]["time"]
        cnt += fast_perf[info]["count"]
    t_average = t_average / len(fast_perf)
    cnt = cnt / len(fast_perf)
    print("Fast Evaluation")
    print(f"Average running time: {t_average}")
    print(f"Average bbox: {cnt}")


def quality_eva(ann_file_path):
    pass


def task_coco_running_time():
    """
    Evaluate the running time of selected samples from COCO dataset.
    """
    ann_file_path = "D:\\Data\\BBH_Exp\\COCO\\instances_trainval2017.json"
    running_time_eva(ann_file_path)


def task_coco_quality_eva():
    ann_file_path = "D:\\Data\\BBH_Exp\\COCO\\instances_trainval2017.json"
    img_dir = "D:\\Data\\BBH_Exp\\COCO\\images"
    out_dir = "D:\\Data\\BBH_Exp\\COCO\\result"

def main():
    task_coco_running_time()


if __name__ == '__main__':
    main()
