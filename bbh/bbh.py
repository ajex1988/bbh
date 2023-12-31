import numpy as np
import copy
import cv2

def uncovered_area_metric(bbox_s, bbox_t):
    x_tl, y_tl = min(bbox_s[0], bbox_t[0]), min(bbox_s[1], bbox_t[1])
    x_br, y_br = max(bbox_s[2], bbox_t[2]), max(bbox_s[3], bbox_t[3])
    area_s = (bbox_s[2]-bbox_s[0])*(bbox_s[3]-bbox_s[1])
    area_t = (bbox_t[2]-bbox_t[0])*(bbox_t[3]-bbox_t[1])
    area_st = (x_br-x_tl)*(y_br-y_tl)
    return area_st - area_s - area_t


class BBoxesVis:
    '''
    Bounding Box Visualization
    h: int, image height
    w: int, image width
    bboxes: list of tuples(list). Each bbox is a 4-element tuple containing xmin, ymin, xmax, ymax
    '''
    def __init__(self, h, w, bboxes):
        self.h = h
        self.w = w
        self.bboxes = bboxes
        self.canvas = np.ones((h, w, 3), dtype=np.uint8)*255

    def _get_colors(self, n_color):
        return [np.random.randint(low=0, high=256, size=(3,)) for _ in range(n_color)]

    def render(self):
        n_bboxes = len(self.bboxes)
        colors = self._get_colors(n_bboxes)
        for color, bbox in zip(colors, self.bboxes):
            xmin, ymin, xmax, ymax = bbox[:]
            self.canvas[ymin:ymax+1, xmin:xmax+1] = color
        return self.canvas


class BBH:
    def __init__(self,
                 bboxes,
                 dist_metric=uncovered_area_metric):
        self.bboxes = bboxes
        self.dist_metric = dist_metric

    def _merge2(self, bbox_s, bbox_t):
        """
        Merge two bboxes and return the merged one
        """
        x_tl, y_tl = min(bbox_s[0], bbox_t[0]), min(bbox_s[1], bbox_t[1])
        x_br, y_br = max(bbox_s[2], bbox_t[2]), max(bbox_s[3], bbox_t[3])
        return [x_tl, y_tl, x_br, y_br]

    def merge(self):
        raise NotImplementedError("Merge method should be implemented")


class BBHNaive(BBH):
    """
    Naive approach for bounding box hierarchy
    """
    def __init__(self,
                 bboxes,
                 dist_metric=uncovered_area_metric):
        super().__init__(bboxes=bboxes, dist_metric=dist_metric)

    def merge(self):
        n = len(self.bboxes)
        hierarchy = [[] for i in range(n)]
        hierarchy[n-1] = copy.deepcopy(self.bboxes)
        # There will be n-1 iterations
        for idx in range(n-2, -1, -1):
            # Calculate the pair distance between any two bboxs in previous round
            candidates = hierarchy[idx+1]
            nc = len(candidates)
            to_merge_idx_s = -1
            to_merge_idx_t = -1
            min_dist = float('inf')
            for i in range(nc):
                for j in range(i+1, nc):
                    dist = self.dist_metric(candidates[i], candidates[j])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge_idx_s = i
                        to_merge_idx_t = j
            # Merge the two bboxes that have the minimum distance
            bbox_merged = self._merge2(candidates[to_merge_idx_s], candidates[to_merge_idx_t])
            # Copy the list, delete the candidate two and insert the merged one
            bbox_list = [candidates[i] for i in range(nc) if i!=to_merge_idx_s and i!=to_merge_idx_t]
            bbox_list.append(bbox_merged)
            hierarchy[idx] = bbox_list
        return hierarchy


class BBHFast(BBH):
    def __init__(self):
        pass

    def merge(self):
        pass


def get_test_case():
    bboxes = [[68, 82, 138, 321],
              [202, 81, 252, 327],
              [261, 81, 308, 327],
              [364, 112, 389, 182],
              [362, 192, 389, 305],
              [404, 98, 421, 317],
              [92, 421, 146, 725],
              [80, 738, 134, 1060],
              [209, 399, 227, 456],
              [233, 399, 250, 444],
              [257, 400, 279, 471],
              [281, 399, 298, 440],
              [286, 446, 303, 458],
              [353, 394, 366, 429]]  # Ymin, Xmin, Yax, and Xmax
    return bboxes


def bbh_naive_test():
    """
    Test the naive bbh algorithm
    """
    img_h = 1200
    img_w = 1200
    bboxes_ori = get_test_case()
    alg = BBHNaive(bboxes=bboxes_ori)
    bboxes_hierarchy = alg.merge()
    ori_vis = BBoxesVis(h=img_h,
                        w=img_w,
                        bboxes=bboxes_ori)
    img_ori = ori_vis.render()
    tgt_vis = BBoxesVis(h=img_h,
                        w=img_w,
                        bboxes=bboxes_hierarchy[1])
    img_tgt = tgt_vis.render()
    cv2.imwrite("ori.png", img_ori)
    cv2.imwrite("tgt.png", img_tgt)


def main():
    bbh_naive_test()


if __name__ == "__main__":
    main()
