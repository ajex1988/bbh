'''
Script used to pre-process coco dataset
The goal is to select images that have No. of BBoxes larger than a threshold.
COCO 2017 Train Stats:
{'1-10': 86117, '10-20': 22390, '20-30': 6633, '30-40': 1610, '40-50': 393, '>50': 123}
COCO 2017 Val Stats:
{'1-10': 3620, '10-20': 943, '20-30': 307, '30-40': 62, '40-50': 14, '>50': 6}
'''
import os
import json


def parse_annotation(ann_file_path):
    '''
    Parse coco annotation file
    return a dict where key is the image id,
    and value is a dict containing  bboxes and image path
    '''
    ann = json.load(open(ann_file_path))
    img_info = {}
    id2info = {}
    for info in ann['images']:
        id2info[info['id']] = {"file_name": info['file_name'],
                               "width": info['width'],
                               "height": info['height']}
    ann_list = ann["annotations"]
    for annotation in ann_list:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        if image_id not in img_info:
            img_info[image_id] = {}
            img_info[image_id]["bbox_list"] = [bbox]
            img_info[image_id]["file_name"] = id2info[image_id]["file_name"]
            img_info[image_id]["width"] = id2info[image_id]["width"]
            img_info[image_id]["height"] = id2info[image_id]["height"]
        else:
            img_info[image_id]["bbox_list"].append(bbox)

    return img_info


def select_subset(img_info, threshold=50):
    img_info_sub = {}
    for info in img_info:
        if len(img_info[info]["bbox_list"]) >= threshold:
            img_info_sub[info] = img_info[info]
    return img_info_sub


def cal_coco_stats(ann_file_path):
    '''
    Cal coco dataset statistics in order to set a threshold.
    '''
    img_info = parse_annotation(ann_file_path)
    cnt = {"1-10": 0,
           "10-20": 0,
           "20-30": 0,
           "30-40": 0,
           "40-50": 0,
           ">50": 0}
    for info in img_info:
        bbox_num = len(img_info[info]["bbox_list"])
        if bbox_num < 10:
            cnt["1-10"] += 1
        elif bbox_num < 20:
            cnt["10-20"] += 1
        elif bbox_num < 30:
            cnt["20-30"] += 1
        elif bbox_num < 40:
            cnt["30-40"] += 1
        elif bbox_num < 50:
            cnt["40-50"] += 1
        else:
            cnt[">50"] += 1
    print(cnt)


def task_cal_coco_stats():
    ann_file_path_train = "D:\\Data\\Segmentation\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json"
    print("COCO 2017 Train Stats:")
    cal_coco_stats(ann_file_path_train)
    ann_file_path_val = "D:\\Data\\Segmentation\\COCO\\annotations_trainval2017\\annotations\\instances_val2017.json"
    print("COCO 2017 Val Stats:")
    cal_coco_stats(ann_file_path_val)


def task_coco_select_sub():
    """
    Select a subset of coco images that have rather large number of bboxes.
    """
    ann_file_path_train = "D:\\Data\\Segmentation\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json"
    ann_file_path_val = "D:\\Data\\Segmentation\\COCO\\annotations_trainval2017\\annotations\\instances_val2017.json"
    out_file = "D:\\Data\\BBH_Exp\\COCO\\instances_trainval2017.json"
    img_info_train = parse_annotation(ann_file_path_train)
    img_info_val = parse_annotation(ann_file_path_val)
    img_info = {**img_info_train, **img_info_val}
    img_info_sub = select_subset(img_info)
    with open(out_file, "w") as f:
        json.dump(img_info_sub, f)


def main():
    task_cal_coco_stats()


if __name__ == "__main__":
    main()
