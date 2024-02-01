from PIL import Image, ImageDraw


def gray(v):
    return tuple(int(v*255) for _ in range(3))


def get_color_dict():
    color_dict = {}
    color_dict["BLACK"] = gray(0)
    color_dict["DARK_GRAY"] = gray(.25)
    color_dict["GRAY"] = gray(.5)
    color_dict["LIGHT_GRAY"] = gray(.75)
    color_dict["WHITE"] = gray(1)
    color_dict["RED"] = (255, 0, 0)
    color_dict["GREEN"] = (0, 255, 0)
    color_dict["BLUE"] = (0, 0, 255)
    color_dict["CYAN"] = (0, 255, 255)
    color_dict["MAGENTA"] = (255, 0, 255)
    color_dict["YELLOW"] =  (255, 255, 0)
    color_dict["BACKGR"] = (245, 245, 87)
    color_dict["SQUARE_COLOR"] = (255, 73, 73)
    color_dict["RECT_COLOR"] = (37, 182, 249)
    return color_dict


def draw_bbox_bdy(draw, bbox, fill=None, outline="DARK_GRAY"):
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                   fill=fill, outline=outline)


def visualize(bbox_list, img_h, img_w, fg_color, bg_color):
    color_dict = get_color_dict()
    if fg_color not in color_dict or bg_color not in color_dict:
        raise Exception("Color not supported")
    image = Image.new("RGB", (img_w, img_h), color_dict[bg_color])
    draw = ImageDraw.Draw(image)
    for bbox in bbox_list:
        draw_bbox_bdy(draw=draw,
                      bbox=bbox,
                      outline=color_dict[fg_color])
    return image


def main():
    """Do a test for visualization"""
    bbox_list = [[68, 82, 138, 321],
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
                 [353, 394, 366, 429]]
    img_h = 512
    img_w = 512
    fg_color = "BACKGR"
    bg_color = "LIGHT_GRAY"
    img = visualize(bbox_list=bbox_list,
                    img_h=img_h,
                    img_w=img_w,
                    fg_color=fg_color,
                    bg_color=bg_color)
    img.save("vis.png")


if __name__ == "__main__":
    main()
