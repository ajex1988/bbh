import os.path
import random
from random import randint

"""
https://stackoverflow.com/questions/4373741/how-can-i-randomly-place-several-non-colliding-rects
"""


class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    @staticmethod
    def from_point(other):
        return Point(other.x, other.y)


class Rect(object):
    def __init__(self, x1, y1, x2, y2):
        minx, maxx = (x1,x2) if x1 < x2 else (x2,x1)
        miny, maxy = (y1,y2) if y1 < y2 else (y2,y1)
        self.min, self.max = Point(minx, miny), Point(maxx, maxy)

    @staticmethod
    def from_points(p1, p2):
        return Rect(p1.x, p1.y, p2.x, p2.y)

    width = property(lambda self: self.max.x - self.min.x)
    height = property(lambda self: self.max.y - self.min.y)

plus_or_minus = lambda v: v * [-1, 1][(randint(0, 100) % 2)]  # equal chance +/-1


def quadsect(rect, factor):
    """ Subdivide given rectangle into four non-overlapping rectangles.
        'factor' is an integer representing the proportion of the width or
        height the deviatation from the center of the rectangle allowed.
    """
    # pick a point in the interior of given rectangle
    w, h = rect.width, rect.height  # cache properties
    center = Point(rect.min.x + (w // 2), rect.min.y + (h // 2))
    rect_tl = Rect(center.x - randint(0, w // factor),
                   center.y - randint(0, h // factor),
                   rect.min.x + randint(0, w // factor),
                   rect.min.y + randint(0, h // factor))
    rect_tr = Rect(center.x + randint(0, w // factor),
                   center.y - randint(0, h // factor),
                   rect.max.x - randint(0, w // factor),
                   rect.min.y + randint(0, h // factor))
    rect_bl = Rect(center.x - randint(0, w // factor),
                   center.y + randint(0, h // factor),
                   rect.min.x + randint(0, w // factor),
                   rect.max.y - randint(0, h // factor))
    rect_br = Rect(center.x + randint(0, w // factor),
                   center.y + randint(0, h // factor),
                   rect.max.x - randint(0, w // factor),
                   rect.max.y - randint(0, h // factor))

    # create rectangles from the interior point and the corners of the outer one
    return [rect_tl,
            rect_tr,
            rect_bl,
            rect_br]


def square_subregion(rect):
    """ Return a square rectangle centered within the given rectangle """
    w, h = rect.width, rect.height  # cache properties
    if w < h:
        offset = (h - w) // 2
        return Rect(rect.min.x, rect.min.y+offset,
                    rect.max.x, rect.min.y+offset+w)
    else:
        offset = (w - h) // 2
        return Rect(rect.min.x+offset, rect.min.y,
                    rect.min.x+offset+h, rect.max.y)


def visualize():
    random.seed()

    NUM_RECTS = 200
    REGION = Rect(0, 0, 1280, 1280)
    # call quadsect() until at least the number of rects wanted has been generated
    rects = [REGION]   # seed output list
    while len(rects) <= NUM_RECTS:
        rects = [subrect for rect in rects
                        for subrect in quadsect(rect, 10)]

    random.shuffle(rects)  # mix them up
    sample = random.sample(rects, NUM_RECTS)  # select the desired number
    print('%d out of the %d rectangles selected' % (NUM_RECTS, len(rects)))

    #################################################
    # extra credit - create an image file showing results

    from PIL import Image, ImageDraw

    def gray(v):
        return tuple(int(v*255) for _ in range(3))

    BLACK, DARK_GRAY, GRAY = gray(0), gray(.25), gray(.5)
    LIGHT_GRAY, WHITE = gray(.75), gray(1)
    RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
    CYAN, MAGENTA, YELLOW = (0, 255, 255), (255, 0, 255), (255, 255, 0)
    BACKGR, SQUARE_COLOR, RECT_COLOR = (245, 245, 87), (255, 73, 73), (37, 182, 249)

    imgx, imgy = REGION.max.x + 1, REGION.max.y + 1
    image = Image.new("RGB", (imgx, imgy), BACKGR)  # create color image
    draw = ImageDraw.Draw(image)

    def draw_rect(rect, fill=None, outline=WHITE):
        draw.rectangle([(rect.min.x, rect.min.y), (rect.max.x, rect.max.y)],
                   fill=fill, outline=outline)


    # then draw the random sample of them selected
    for rect in sample:
        draw_rect(rect, fill=RECT_COLOR, outline=WHITE)

    filename = 'square_quadsections.png'
    image.save(filename, "PNG")
    print(repr(filename), 'output image saved')


def generate_rects(sample_num, rect_num, img_h, img_w, out_dir):
    """
    Generate randon rects and save the results
    """
    random.seed()
    NUM_RECTS = rect_num
    REGION = Rect(0, 0, img_w, img_h)

    # Prepare output dir
    out_rect_dir = os.path.join(out_dir, "rect")
    if not os.path.exists(out_rect_dir):
        os.makedirs(out_rect_dir)
    out_vis_dir = os.path.join(out_dir, "vis")
    if not os.path.exists(out_vis_dir):
        os.makedirs(out_vis_dir)

    # Generate sample_num samples
    for i in range(sample_num):
        out_sample_id = f"{rect_num}_{i}"
        out_txt_name = out_sample_id + ".txt"
        out_txt_path = os.path.join(out_rect_dir, out_txt_name)

        out_img_name = out_sample_id + ".png"
        out_img_path = os.path.join(out_vis_dir, out_img_name)

        # call quadsect() until at least the number of rects wanted has been generated
        rects = [REGION]  # seed output list
        while len(rects) <= NUM_RECTS:
            rects = [subrect for rect in rects
                    for subrect in quadsect(rect, 10)]

        random.shuffle(rects)  # mix them up
        sample = random.sample(rects, NUM_RECTS)  # select the desired number
        print('%d out of the %d rectangles selected' % (NUM_RECTS, len(rects)))

        # Save to txt
        with open(out_txt_path, 'w') as writer:
            for j in range(rect_num):
                x1, y1, x2, y2 = rects[j].min.x, rects[j].min.y, rects[j].max.x, rects[j].max.y
                writer.write(f"{x1} {y1} {x2} {y2}\n")

        #################################################
        # extra credit - create an image file showing results
        from PIL import Image, ImageDraw
        def gray(v):
            return tuple(int(v * 255) for _ in range(3))

        BLACK, DARK_GRAY, GRAY = gray(0), gray(.25), gray(.5)
        LIGHT_GRAY, WHITE = gray(.75), gray(1)
        RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        CYAN, MAGENTA, YELLOW = (0, 255, 255), (255, 0, 255), (255, 255, 0)
        BACKGR, SQUARE_COLOR, RECT_COLOR = (245, 245, 87), (255, 73, 73), (37, 182, 249)

        imgx, imgy = REGION.max.x + 1, REGION.max.y + 1
        image = Image.new("RGB", (imgx, imgy), BACKGR)  # create color image
        draw = ImageDraw.Draw(image)

        def draw_rect(rect, fill=None, outline=WHITE):
            draw.rectangle([(rect.min.x, rect.min.y), (rect.max.x, rect.max.y)],
                        fill=fill, outline=outline)



        # draw the random sample of them selected
        for rect in sample:
            draw_rect(rect, fill=RECT_COLOR, outline=WHITE)

        image.save(out_img_path, "PNG")
        print(repr(out_img_path), 'output image saved')


def task_test_ori_alg():
    """
    Test the original algorithm pulled from github
    """
    visualize()


def task_generate_exp_data():
    """
    Generate the test samples to use in the experiment
    """
    out_dir_p = "D:\\tmp\\bbh\\test_case"
    sample_num = 100
    rect_num_list = [10, 100, 1000, 10000, 100000, 1000000]
    img_size_list = [[256, 256],
                     [512, 512],
                     [1024, 1024],
                     [2048, 2048],
                     [8192, 8192],
                     [32768, 32768]]
    for i in range(len(rect_num_list)):
        rect_num = rect_num_list[i]
        h, w = img_size_list[i][0], img_size_list[i][1]
        out_dir = os.path.join(out_dir_p, f"{rect_num}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        generate_rects(sample_num=sample_num,
                       rect_num=rect_num,
                       img_h=h,
                       img_w=w,
                       out_dir=out_dir)


def main():
    #task_test_ori_alg()
    task_generate_exp_data()


if __name__ == "__main__":
    main()
