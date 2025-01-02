import os


def parse_cityperson_annotation(ann_dir):
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
    for city_name in city_name_list:
        city_dir = os.path.join(ann_dir, city_name)
        ann_file_list = os.listdir(city_dir)
    return info


def filter_samples():
    """
    Filter samples according to the number of samples per category.
    :return:
    """
    pass


def main():
    pass


if __name__ == "__main__":
    main()
