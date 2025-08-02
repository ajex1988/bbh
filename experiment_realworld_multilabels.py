import os
import json
from PIL import Image, ImageDraw
from bbh.bbh import BBHFastMultiLabel

def name2label_dict():
    dict = {}
    dict["ignore"] = 0
    dict["pedestrian"] = 1
    dict["rider"] = 2
    dict["sitting person"] = 3
    dict["person (other)"] = 4
    dict["person group"] = 5
    return dict

def label2color_dict():
    dict = {}
    dict[0] = (255, 255, 0)
    dict[1] = (255, 0, 0)
    dict[2] = (0, 255, 0)
    dict[3] = (0, 0, 255)
    dict[4] = (0, 255, 255)
    dict[5] = (255, 0, 255)
    return dict

def task_visualize_cityperson_multilabel():
    ann_file_path = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation_filtered.json"
    img_dir = "D:\\Data\\BBH_Exp\\CityPersons\\city_persons\\images"
    out_dir = "D:\\Data\\BBH_Exp\\CityPersons\\multi_labels\\results"

    select_list = [-50, -40, -30, -20, -10, -5]

    name2label = name2label_dict()
    label2color = label2color_dict()

    with open(ann_file_path, 'r') as f:
        info = json.load(f)

    for fid in info:
        city_name, no_1, no_2, gtbox_name = fid.split('_')
        img_name = f"{city_name}_{city_name}_{no_1}_{no_2}_leftImg8bit.png"
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')

        bboxes = []
        labels = []

        labeled_bboxes = info[fid]
        for lname in labeled_bboxes:
            bbox = labeled_bboxes[lname]
            bboxes.extend(bbox)
            labels.extend([name2label[lname]]*len(bbox))

        alg = BBHFastMultiLabel(bboxes=bboxes, labels=labels)
        bbh, bbh_label = alg.merge()

        for i in select_list:
            bbox_list = bbh[i]
            bbox_label_list = bbh_label[i]

            image = img.copy()
            draw = ImageDraw.Draw(image)
            for j, bbox in enumerate(bbox_list):
                color = label2color[bbox_label_list[j]]
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                               fill=None, outline=color)
            image.save(os.path.join(out_dir, f"{img_name[:-4]}_{abs(i)}.png"))

def task_visualize_cityperson_ori():
    """
    Draw the original bounding boxes with class labels
    """
    ann_file_path = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation_filtered.json"
    img_dir = "D:\\Data\\BBH_Exp\\CityPersons\\city_persons\\images"
    out_dir = "D:\\Data\\BBH_Exp\\CityPersons\\multi_labels\\results_ori"

    select_list = [0]

    name2label = name2label_dict()
    label2color = label2color_dict()

    with open(ann_file_path, 'r') as f:
        info = json.load(f)

    for fid in info:
        city_name, no_1, no_2, gtbox_name = fid.split('_')
        img_name = f"{city_name}_{city_name}_{no_1}_{no_2}_leftImg8bit.png"
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')

        bboxes = []
        labels = []

        labeled_bboxes = info[fid]
        for lname in labeled_bboxes:
            bbox = labeled_bboxes[lname]
            bboxes.extend(bbox)
            labels.extend([name2label[lname]] * len(bbox))

        alg = BBHFastMultiLabel(bboxes=bboxes, labels=labels)
        bbh, bbh_label = alg.merge()

        for i in select_list:
            bbox_list = bbh[i]
            bbox_label_list = bbh_label[i]

            image = img.copy()
            draw = ImageDraw.Draw(image)
            for j, bbox in enumerate(bbox_list):
                color = label2color[bbox_label_list[j]]
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                               fill=None, outline=color)
            image.save(os.path.join(out_dir, f"{img_name[:-4]}_{abs(i)}.png"))

def main():
    task_visualize_cityperson_multilabel()
    # task_visualize_cityperson_ori()


if __name__ == '__main__':
    main()
