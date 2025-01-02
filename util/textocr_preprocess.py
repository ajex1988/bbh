import os
import json
import shutil


def parse_annotation(ann_file_path, img_dir, out_dir, threshold=50):
    img_info = {}
    with open(ann_file_path, 'r') as f:
        data = json.load(f)
    for ann_id in data['anns']:
        img_id = data['anns'][ann_id]['image_id']
        if data['imgs'][img_id]['set'] != "train":
            continue
        bbox = data['anns'][ann_id]['bbox']
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
        if img_id not in img_info:
            img_info[img_id] = {'bbox_list': [bbox],
                                'file_name': data["imgs"][img_id]['file_name'],
                                'width': data["imgs"][img_id]['width'],
                                'height': data["imgs"][img_id]['height']}
        else:
            img_info[img_id]['bbox_list'].append(bbox)
    img_info_select = {}
    # Select the subset that contains >= threshold bboxes
    for img_id in img_info:
        if len(img_info[img_id]['bbox_list']) > threshold:
            img_info_select[img_id] = img_info[img_id]
            file_name = img_info[img_id]['file_name'].split('/')[-1]
            img_info_select[img_id]['file_name'] = file_name
            img_pth_src = os.path.join(img_dir, file_name)
            img_path_dest = os.path.join(out_dir, file_name)
            shutil.copyfile(img_pth_src, img_path_dest)
    with open(os.path.join(out_dir, "textocr.json"), 'w') as f:
        json.dump(img_info_select, f)


def cal_textocr_stats(ann_file_path):
    '''
    Cal textocr dataset statistics in order to set a threshold.
    '''
    with open(ann_file_path, 'r') as f:
        data = json.load(f)
    img_info = data
    cnt = {"1-100": 0,
           "100-200": 0,
           "200-300": 0,
           "300-400": 0,
           "400-500": 0,
           ">500": 0}
    for info in img_info:
        bbox_num = len(img_info[info]["bbox_list"])
        if bbox_num < 100:
            cnt["1-100"] += 1
        elif bbox_num < 200:
            cnt["100-200"] += 1
        elif bbox_num < 300:
            cnt["200-300"] += 1
        elif bbox_num < 400:
            cnt["300-400"] += 1
        elif bbox_num < 500:
            cnt["400-500"] += 1
        else:
            cnt[">500"] += 1
    print(cnt)


def task_cal_stats():
    """
    {'1-100': 2656, '100-200': 1693, '200-300': 446, '300-400': 203, '400-500': 88, '>500': 157}
    """
    ann_file_path = r"D:\Data\BBH_Exp\TextOCR\textocr.json"
    cal_textocr_stats(ann_file_path=ann_file_path)


def task_parse_textocr():
    ann_file_path = "D:\\Data\\BBH_Exp\\TextOCR\\train\\TextOCR_0.1_train.json"
    img_dir = "D:\\Data\\BBH_Exp\\TextOCR\\train\\train_val_images\\train_images"
    out_dir = "D:\\Data\\BBH_Exp\\TextOCR_500\\images"
    parse_annotation(ann_file_path, img_dir, out_dir, threshold=500)


def main():
    #task_cal_stats()
    task_parse_textocr()


if __name__ == '__main__':
    main()