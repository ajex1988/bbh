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
            file_name = img_info[img_id]['file_name'].split('/')[1]
            img_pth_src = os.path.join(img_dir, file_name)
            img_path_dest = os.path.join(out_dir, file_name)
            shutil.copyfile(img_pth_src, img_path_dest)
    with open(os.path.join(out_dir, "textocr.json"), 'w') as f:
        json.dump(img_info_select, f)


def task_parse_textocr():
    ann_file_path = "D:\\Data\\BBH_Exp\\TextOCR\\train\\TextOCR_0.1_train.json"
    img_dir = "D:\\Data\\BBH_Exp\\TextOCR\\train\\train_val_images\\train_images"
    out_dir = "D:\\Data\\BBH_Exp\\TextOCR\\images"
    parse_annotation(ann_file_path, img_dir, out_dir)


def main():
    task_parse_textocr()


if __name__ == '__main__':
    main()