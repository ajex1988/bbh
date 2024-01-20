import os
import time
from util.extract_bbox_from_real_cases import load_bbox_from_txt
import logging
from bbh.bbh import BBHNaive, BBHFast


def exp_homepage_case_test():
    """
    Test the homepage case
    """
    bbox_file_path = "D:/tmp/20240117/home_menu_det_res/2K_Youtube_00015/preds/2K_Youtube_00015.txt"
    log_txt = "D:\\tmp\\bbh\\exp\\homepage_test\\log.txt"
    logging.basicConfig(filename=log_txt, encoding='utf-8', level=logging.INFO)

    bbox_list = load_bbox_from_txt(txt_file_path=bbox_file_path)
    bbox_num = len(bbox_list)
    logging.info(f'Number of BBox: {bbox_num}')

    alg_naive = BBHNaive(bboxes=bbox_list)
    t_naive_start = time.perf_counter()
    h_naive = alg_naive.merge()
    t_naive_end = time.perf_counter()
    logging.info(f"Naive approach: {t_naive_end-t_naive_start}")

    alg_fast = BBHFast(bboxes=bbox_list)
    t_fast_start = time.perf_counter()
    h_fast = alg_fast.merge()
    t_fast_end = time.perf_counter()
    logging.info(f"Fast approach: {t_fast_end-t_fast_start}")


def main():
    exp_homepage_case_test()


if __name__ == "__main__":
    main()
