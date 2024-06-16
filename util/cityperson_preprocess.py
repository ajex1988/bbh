import os
import json
import shutil

import cv2


def parse_annotation(ann_dir, img_dir, out_dir, threshold=50):
    """
    city person has the same structure as cityscape
    -city1
        --file1
        --file2
        ...
    -city2
    -...
    """
    city_name_list = os.listdir(ann_dir)
    img_info = {}
    for city_name in city_name_list:
        city_dir = os.path.join(ann_dir, city_name)
        ann_file_list = os.listdir(city_dir)
        for ann_file_name in ann_file_list:
            ann_file_path = os.path.join(city_dir, ann_file_name)
            with open(ann_file_path, 'r') as f:
                ann_data = json.load(f)
            obj_num = len(ann_data['objects'])
            if obj_num >= threshold:
                bbox_list = []
                for obj_info in ann_data['objects']:
                    bbox = obj_info['bbox']
                    bbox_list.append((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
                # Copy the image
                img_id = ann_file_name[:-24]
                img_name = f"{img_id}_leftImg8bit.png"
                img_file_path = os.path.join(img_dir, city_name, img_name)
                img_file_path_dest = os.path.join(out_dir, "images", f"{city_name}_{img_name}")
                shutil.copy(img_file_path, img_file_path_dest)

                # Set the json annotation
                ann_id = f"{city_name}_{img_id}"
                img_info[ann_id] = {"bbox_list": bbox_list,
                                    "file_name": f"{city_name}_{img_name}",
                                    "width": ann_data['imgWidth'],
                                    "height": ann_data['imgHeight']}
    with open(os.path.join(out_dir, 'city_person.json'), 'w') as f:
        json.dump(img_info, f)


def task_parse_cityperson():
    ann_dir = "D:\\Data\\BBH_Exp\\CityPersons\\gtBbox_cityPersons_trainval\\gtBboxCityPersons\\train"
    img_dir = "D:\\Data\\Segmentation\\cityscape\\leftImg8bit_trainvaltest\\leftImg8bit\\train"
    out_dir = "D:\\Data\\BBH_Exp\\CityPersons\\city_persons"
    parse_annotation(ann_dir=ann_dir,
                     img_dir=img_dir,
                     out_dir=out_dir)


def main():
    pass


if __name__ == "__main__":
    main()
