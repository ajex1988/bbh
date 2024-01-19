import cv2
import json
import numpy as np


def cv2_mask2bbox(mask_path, txt_path):
    """
    Given a mask image path, output a bbox text file in the following format:
    img_width img_height
    x1 y1 x2 y2
    ...
    xn1 xn2 yn1 yn2
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape[:2]
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    with open(txt_path, 'w') as writer:
        writer.write(f"{w} {h}\n")
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x+w, y+h
            writer.write(f"{x1} {y1} {x2} {y2}\n")


def task_extract_homepage_bboxes():
    """
    Extract the bounding boxes from homepage image text detection results
    """
    mask_path = "D:\\tmp\\20230810\\bbox_mask_txtdet_bin\\2K_Youtube_00015.png"
    txt_path = "D:\\tmp\\20240117\\debug.txt"
    cv2_mask2bbox(mask_path=mask_path,
                  txt_path=txt_path)


def parse_from_mmocr_det(in_json_path, out_txt_path):
    """
    Parse the detection from mmocr detection results
    """
    with open(in_json_path, 'r') as f:
        det_res = json.load(f)
    with open(out_txt_path, 'w') as writer:
        polygons = det_res["det_polygons"] # Here we use det_polygon which is common in mmocr detection results
        for cnt in polygons:
            cnt = np.float32(np.array(cnt))
            cnt = np.reshape(cnt, (-1,1,2))
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x+w, y+h
            writer.write(f"{x1} {y1} {x2} {y2}\n")


def load_bbox_from_txt(txt_file_path):
    """
    Read the txt file and return the bbox
    The parsed bbox has the following format:
        [[x1, y1, x2, y2],...[xn1, yn1, xn2, yn2]]
    """
    with open(txt_file_path, 'r') as reader:
        lines = reader.readlines()
    bbox_list = []
    for line in lines:
        bbox = line.split()
        bbox = [int(b) for b in bbox]
        bbox_list.append(bbox)
    return bbox_list


def visualize_bbox(img_h, img_w, bbox_list):
    """
    Visualize the bbox by generating a mask image, where 0 indicts background and 255 indicates foreground
    """
    img = np.zeros((img_h, img_w), dtype=np.uint8)
    for bbox in bbox_list:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        img[y1:y2, x1:x2] = 255
    return img


def task_test_parse_from_mmocr_det():
    in_json_path = "D:/tmp/20240117/home_menu_det_res/2K_Youtube_00015/preds/2K_Youtube_00015.json"
    out_txt_path = "D:/tmp/20240117/home_menu_det_res/2K_Youtube_00015/preds/2K_Youtube_00015.txt"
    parse_from_mmocr_det(in_json_path=in_json_path,
                         out_txt_path=out_txt_path)


def task_verify_extracted_bbox():
    """
    Verify the extracted bbox by rendering them in an image
    """
    bbox_file_path = "D:/tmp/20240117/home_menu_det_res/2K_Youtube_00015/preds/2K_Youtube_00015.txt"
    out_img_path = "D:/tmp/20240117/home_menu_det_res/2K_Youtube_00015/preds/2K_Youtube_00015_rendered.png"
    img_h = 1080
    img_w = 1920

    # Load bbox
    bbox_list = load_bbox_from_txt(bbox_file_path)
    rendered_mask = visualize_bbox(img_h=img_h,
                                   img_w=img_w,
                                   bbox_list=bbox_list)
    cv2.imwrite(out_img_path, rendered_mask)


def main():
    #task_extract_homepage_bboxes()
    #task_test_parse_from_mmocr_det()
    task_verify_extracted_bbox()


if __name__ == "__main__":
    main()
