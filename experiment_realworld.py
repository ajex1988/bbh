import os
import sys
import time
from util.extract_bbox_from_real_cases import load_bbox_from_txt
from util.evaluate_acc import cal_iou_bbox_list
import logging
from bbh.bbh import BBHNaive, BBHFast
from util.visualization import visualize_realworld
from PIL import Image
import json
import statistics

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


def quality_eva(ann_file_path, img_dir, out_dir, h_levels=50):
    vis_dir = os.path.join(out_dir, "vis")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    eva_info = {"fast_vs_gt":[],
                "bf_vs_gt":[],
                "fast_vs_bf":[]}
    fast_vs_gt_list = {}
    bf_vs_gt_list = {}
    fast_vs_bf_list = {}
    with open(ann_file_path, 'r') as f:
        img_info = json.load(f)
    for info in img_info:
        img_name = img_info[info]['name']
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')
        bbox_list = img_info[info]['bbox_list']
        bbox_list_gt = bbox_list
        img_h = img_info[info]['height']
        img_w = img_info[info]['width']
        alg_bf = BBHNaive(bboxes=bbox_list)
        alg_fast = BBHFast(bboxes=bbox_list)
        bbh_fast = alg_fast.merge()
        bbh_bf = alg_bf.merge()

        for i in range(h_levels):
            bbox_list_fast = bbh_fast[i]
            bbox_list_bf = bbh_bf[i]
            fast_vs_gt = cal_iou_bbox_list(bbox_list_src=bbox_list_fast,
                                           bbox_list_tgt=bbox_list_gt,
                                           height=img_h,
                                           width=img_w)
            bf_vs_gt = cal_iou_bbox_list(bbox_list_src=bbox_list_bf,
                                         bbox_list_tgt=bbox_list_gt,
                                         height=img_h,
                                         width=img_w)
            fast_vs_bf = cal_iou_bbox_list(bbox_list_src=bbox_list_fast,
                                           bbox_list_tgt=bbox_list_bf,
                                           height=img_h,
                                           width=img_w)
            if fast_vs_gt_list[i]:
                fast_vs_gt_list[i].append(fast_vs_gt)
            else:
                fast_vs_gt_list[i] = [fast_vs_gt]
            if bf_vs_gt_list:
                bf_vs_gt_list.append(bf_vs_gt)
            else:
                bf_vs_gt_list = [bf_vs_gt]
            if fast_vs_bf_list:
                fast_vs_bf_list.append(fast_vs_bf)
            else:
                fast_vs_bf_list = [fast_vs_bf]

            img_vis_gt = visualize_realworld(bbox_list=bbox_list_gt,
                                             image=img,
                                             bbox_color="BLUE")
            img_vis_gt.save(os.path.join(vis_dir, f"{img_name[:-4]}_gt_{i}.png"))

            img_vis_fast = visualize_realworld(bbox_list=bbox_list_fast,
                                               image=img,
                                               bbox_color="CYAN")
            img_vis_fast.save(os.path.join(vis_dir, f"{img_name[:-4]}_fast_{i}.png"))

            img_vis_bf = visualize_realworld(bbox_list=bbox_list_bf,
                                             image=img,
                                             bbox_color="MAGENTA")
            img_vis_bf.save(os.path.join(vis_dir, f"{img_name[:-4]}_bf_{i}.png"))

    for i in range(h_levels):
        eva_info["fast_vs_gt"].append(statistics.mean(fast_vs_gt_list[i]))
        eva_info["bf_vs_gt"].append(statistics.mean(bf_vs_gt_list[i]))
        eva_info["fast_vs_bf"].append(statistics.mean(fast_vs_bf_list[i]))
    print("Quality Evaluation")
    print(eva_info)
    with open(os.path.join(out_dir, "eva_info.json"), 'w') as f:
        json.dump(eva_info, f)


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
    quality_eva(ann_file_path=ann_file_path,
                img_dir=img_dir,
                out_dir=out_dir)

def main():
    task_coco_running_time()


if __name__ == '__main__':
    main()
