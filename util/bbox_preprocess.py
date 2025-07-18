
def _overlap(bbox_s, bbox_t):
    # Check if two bboxs overlap
    xmin = max(bbox_s[0], bbox_t[0])
    ymin = max(bbox_s[1], bbox_t[1])
    xmax = min(bbox_s[2], bbox_t[2])
    ymax = min(bbox_s[3], bbox_t[3])

    if xmax > xmin and ymax > ymin:
        return True
    else:
        return False

def remove_overlap_bboxes(bboxes):
    """
    Remove the overlapping bounding boxes
    """
    _bboxes = bboxes.copy()
    _bboxes_sorted = sorted(_bboxes, key=lambda bbox: abs(bbox[2] - bbox[0])*abs(bbox[3] - bbox[1]), reverse=True)
    bboxes_nonoverlap = []
    for i in range(len(_bboxes_sorted)):
        bbox = _bboxes_sorted[i]
        has_overlap = False
        for j in range(i+1, len(_bboxes_sorted)):
            if _overlap(bbox, _bboxes_sorted[j]):
                has_overlap = True
                break
        if not has_overlap:
            bboxes_nonoverlap.append(_bboxes_sorted[i])
    return bboxes_nonoverlap

def main():
    """
    Test the func that removes overlap bboxes
    """
    bboxes = [[10, 10, 100, 100],
              [20, 20, 30, 30],
              [70, 20, 80, 30],
              [20, 70, 30, 80],
              [70, 70, 80, 80],
              [5, 75, 85, 85]]
    bboxes_nonoverlap = remove_overlap_bboxes(bboxes)
    print(bboxes_nonoverlap)


if __name__ == "__main__":
    main()
