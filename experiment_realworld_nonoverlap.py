import os
import json
from PIL import Image, ImageDraw
from bbh.bbh import BBHFastNonOverlap, BBHFast


def task_vis_on_city_person():
    ann_file_path = r"D:\Data\BBH_Exp\CityPersons\multi_labels\annotation_filtered.json"
    img_dir = "D:\\Data\\BBH_Exp\\CityPersons\\city_persons\\images"
    out_dir = "D:\\Data\\BBH_Exp\\CityPersons\\non_overlap\\results"

    os.makedirs(out_dir, exist_ok=True)

    select_list = [-20, -10, -5]
    color_nonoverlap = (255, 255, 255)
    color_ori = (0, 255, 0)

    with open(ann_file_path, 'r') as f:
        info = json.load(f)

    for fid in info:
        city_name, no_1, no_2, gtbox_name = fid.split('_')
        img_name = f"{city_name}_{city_name}_{no_1}_{no_2}_leftImg8bit.png"
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert('RGB')

        bboxes = []

        labeled_bboxes = info[fid]
        for lname in labeled_bboxes:
            bbox = labeled_bboxes[lname]
            bboxes.extend(bbox)

        alg_fast = BBHFast(bboxes=bboxes)
        alg_nonoverlap = BBHFastNonOverlap(bboxes=bboxes)
        bbh_fast = alg_fast.merge()
        bbh_nonoverlap = alg_nonoverlap.merge()

        for i in select_list:
            bbox_list = bbh_fast[i]
            bbox_nonoverlap_list = bbh_nonoverlap[i]

            image = img.copy()
            draw = ImageDraw.Draw(image)
            for j, bbox in enumerate(bbox_list):
                draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                               fill=None, outline=color_ori)
            image.save(os.path.join(out_dir, f"{img_name[:-4]}_{abs(i)}_ori.png"))

            image_nonoverlap = img.copy()
            draw_nonoverlap = ImageDraw.Draw(image_nonoverlap)
            for j, bbox in enumerate(bbox_nonoverlap_list):
                draw_nonoverlap.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                                          fill=None, outline=color_nonoverlap)
            image_nonoverlap.save(os.path.join(out_dir, f"{img_name[:-4]}_{abs(i)}_nonoverlap.png"))


def main():
    task_vis_on_city_person()


if __name__ == "__main__":
    main()
