import numpy as np


def cal_iou_bool(mask_src, mask_tgt):
    """
    Calculate the IoU between two bool images
    """
    mask_intersection = np.logical_and(mask_src, mask_tgt)

    area_src = np.count_nonzero(mask_src)
    area_tgt = np.count_nonzero(mask_tgt)
    area_intersection = np.count_nonzero(mask_intersection)

    iou = area_intersection / (area_src + area_tgt - area_intersection)
    return iou


def cal_iou_bbox_list(bbox_list_src, bbox_list_tgt, height, width):
    """
    Calculate the IoU between 2 bounding box list
    """
    mask_src = np.zeros((height, width), dtype=bool)
    mask_tgt = np.zeros((height, width), dtype=bool)

    for bbox in bbox_list_src:
        x1, y1, x2, y2 = bbox[:]
        mask_src[y1:y2+1, x1:x2+1] = 1

    for bbox in bbox_list_tgt:
        x1, y1, x2, y2 = bbox[:]
        mask_tgt[y1:y2+1, x1:x2+1] = 1

    iou = cal_iou_bool(mask_src=mask_src, mask_tgt=mask_tgt)
    return iou


def main():
    bbox_list_src = [[10, 10, 19, 19]]
    bbox_list_tgt = [[15, 10, 24, 19],
                     [40, 40, 54, 49]]
    height = 120
    width = 120
    iou_gt = 1/6
    iou = cal_iou_bbox_list(bbox_list_src=bbox_list_src,
                            bbox_list_tgt=bbox_list_tgt,
                            height=height,
                            width=width)
    print(f"Ground truth IoU: {iou_gt}")
    print(f"Calculated IoU: {iou}")


if __name__ == "__main__":
    main()
