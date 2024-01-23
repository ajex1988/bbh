import os
import sys
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


def running_time_test_by_rect_num(in_dir, out_path, alg_name='FAST'):
    """
    Calculate the running time of a particular rect_num
    This will run sample_num time and calculate the average
    Algorithm name can be either BF or FAST
    """
    logging.basicConfig(filename=out_path, encoding='utf-8', level=logging.INFO)
    sample_file_list = os.listdir(in_dir)
    t_list = []
    for sample_file_name in sample_file_list:
        sample_file_path = os.path.join(in_dir, sample_file_name)
        bbox_list = load_bbox_from_txt(txt_file_path=sample_file_path)

        if alg_name == "FAST":
            alg = BBHFast(bboxes=bbox_list)
        elif alg_name == "BF":
            alg = BBHNaive(bboxes=bbox_list)
        else:
            raise Exception("Unknown Algorithm")
        t_start = time.perf_counter()
        _ = alg.merge()
        t_end = time.perf_counter()
        t_cost = t_end - t_start
        t_list.append(t_cost)

        logging.info(f'Sample {sample_file_name} cost: {t_cost}')
    t_avg = sum(t_list) / len(t_list)
    logging.info(f"Avg: {t_avg}")


def task_running_time_exp():
    in_dir = sys.argv[1]
    out_path = sys.argv[2]
    alg_name = sys.argv[3]
    running_time_test_by_rect_num(in_dir=in_dir,
                                  out_path=out_path,
                                  alg_name=alg_name)


def main():
    #exp_homepage_case_test()
    task_running_time_exp()

if __name__ == "__main__":
    main()
