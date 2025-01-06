import os
import json
import tqdm


def parse_cityperson_annotation(ann_dir, total_obj_num_thre=50):
    """
    Parse the annotation files of the cityperson dataset.
    For this dataset, each image has a corresponding annotation file, and images are organized by cities. For example:
    train
        aachen
            img1
            img2
            ...
        bremen
            img1
            img2
            ...
        ...
        zuich
            img1
            img2
            ...
    Return the parsed annotations.
    """
    city_name_list = os.listdir(ann_dir)
    info = {}
    for city_name in tqdm.tqdm(city_name_list):
        city_dir = os.path.join(ann_dir, city_name)
        ann_file_list = os.listdir(city_dir)
        for ann_file_name in ann_file_list:
            ann_file_path = os.path.join(city_dir, ann_file_name)
            with open(ann_file_path, 'r') as f:
                ann_data = json.load(f)
            obj_num = len(ann_data['objects'])
            if obj_num >= total_obj_num_thre:
                objects = ann_data['objects']
                sample_info = {}
                for obj in objects:
                    label = obj['label']
                    if label not in sample_info:
                        # Convert to [x1, y1, x2, y2] from [x,y,w,h]
                        sample_info[label] = [[obj["bbox"][0], obj["bbox"][1],
                                              obj["bbox"][2]+obj["bbox"][0], obj["bbox"][3]+obj["bbox"][1]]]
                    else:
                        sample_info[label].append([obj["bbox"][0], obj["bbox"][1],
                                              obj["bbox"][2]+obj["bbox"][0], obj["bbox"][3]+obj["bbox"][1]])
                file_id = os.path.splitext(ann_file_name)[0]
                info[file_id] = sample_info

    return info


def filter_samples(ann_path, num_classes=2, second_thre=5):
    """
    Filter samples according to the number of samples per category.
    """
    with open(ann_path, 'r') as reader:
        info = json.load(reader)
    filtered_info = {}
    for file_id, sample_info in info.items():
        num_bboxes_list = [len(item) for item in sample_info.values()]
        num_bboxes_list_sorted = sorted(num_bboxes_list, reverse=True)
        if len(num_bboxes_list_sorted) >= num_classes and num_bboxes_list_sorted[1] >= second_thre:
            filtered_info[file_id] = sample_info
            print(file_id)
    print(f"{len(filtered_info)} samples were kept after filtering.")
    return filtered_info


def task_parse_all_and_save_as_one():
    """
    Parse all the annotation files of the cityperson dataset, and save as one annotation file.
    """
    ann_dir = "D:\\Data\\BBH_Exp\\CityPersons\\gtBbox_cityPersons_trainval\\gtBboxCityPersons\\train"
    tgt_file = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation.json"
    info = parse_cityperson_annotation(ann_dir=ann_dir)
    with open(tgt_file, 'w') as writer:
        json.dump(info, writer)


def task_filter4multilabels():
    in_json_file_path = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation.json"
    out_json_file_path = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation_filtered.json"
    num_classes = 2
    second_thre = 5
    filtered_info = filter_samples(in_json_file_path, num_classes=num_classes, second_thre=second_thre)
    with open(out_json_file_path, 'w') as writer:
        json.dump(filtered_info, writer)


def main():
    task_parse_all_and_save_as_one()
    task_filter4multilabels()


if __name__ == "__main__":
    main()
