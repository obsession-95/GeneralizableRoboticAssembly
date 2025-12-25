# coding: utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robot_control import sim_control
import os


def get_real_img():
    pic_path_x = "./data/real_plug_x.jpg"
    pic_path_y = "./data/real_plug_y.jpg"
    img_x = cv2.imread(pic_path_x)
    img_y = cv2.imread(pic_path_y)

    print(f'img_x.shape = {img_x.shape}')
    print(f'img_y.shape = {img_y.shape}')

    clip_param_x = [320, 660, 640, 980]
    img_x = img_x[clip_param_x[0]:clip_param_x[1], clip_param_x[2]:clip_param_x[3]]
    resized_img_x = cv2.resize(img_x, (img_x.shape[1] // 4, img_x.shape[0] // 4))

    clip_param_y = [250, 420, 520, 690]
    img_y = img_y[clip_param_y[0]:clip_param_y[1], clip_param_y[2]:clip_param_y[3]]
    resized_img_y = cv2.resize(img_y, (img_y.shape[1] // 2, img_y.shape[0] // 2))



    print(resized_img_x.shape, resized_img_y.shape)


    cv2.imshow("original_x", resized_img_x)
    cv2.imshow("original_y", resized_img_y)


    # 二值化
    gray_img_x = cv2.cvtColor(resized_img_x, cv2.COLOR_BGR2GRAY)
    gray_img_y = cv2.cvtColor(resized_img_y, cv2.COLOR_BGR2GRAY)

    # 或者使用CLAHE (Contrast Limited Adaptive Histogram Equalization)

    clahe_x = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    clahe_img_x = clahe_x.apply(gray_img_x)
    clahe_y = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    clahe_img_y = clahe_y.apply(gray_img_y)

    # 显示原始灰度图像和对比度增强后的图像
    # cv2.imshow('CLAHE Image', clahe_image)


    _, bw_img_x = cv2.threshold(clahe_img_x, 60, 255, cv2.THRESH_BINARY)
    _, bw_img_y = cv2.threshold(clahe_img_y, 60, 255, cv2.THRESH_BINARY)

    # 计算总的像素数量
    total_pixels_x = bw_img_x.size
    total_pixels_y = bw_img_y.size

    # 计算非零像素（白色）的数量
    white_pixel_count_x = cv2.countNonZero(bw_img_x)
    white_pixel_count_y = cv2.countNonZero(bw_img_y)

    # 计算零像素（黑色）的数量
    black_pixel_count_x = total_pixels_x - white_pixel_count_x
    black_pixel_count_y = total_pixels_y - white_pixel_count_y

    print(f"x 黑色部分的面积（以像素为单位）: {black_pixel_count_x}")
    print(f"y 黑色部分的面积（以像素为单位）: {black_pixel_count_y}")


    cv2.imshow("binary_img_x", bw_img_x)
    cv2.imshow("binary_img_y", bw_img_y)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_sim_area():
    rob = sim_control.UR5_Sim()
    clip_param = [75, 160, 70, 190]
    img = rob.getVisionImage('Camera_y', clip_param)
    cv2.imshow("original", img)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, bw_img = cv2.threshold(grey_img, 100, 255, cv2.THRESH_BINARY)

    # 计算总的像素数量
    total_pixels = bw_img.size

    # 计算非零像素（白色）的数量
    white_pixel_count = cv2.countNonZero(bw_img)

    # 计算零像素（黑色）的数量
    black_pixel_count = total_pixels - white_pixel_count
    print(f"黑色部分的面积（以像素为单位）: {black_pixel_count}")
    cv2.imshow("binary_img", bw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_sim_img():
    rob = sim_control.UR5_Sim()
    clip_dict = [115, 200, 70, 190]
    img = rob.getVisionImage('Camera_x', clip_dict)
    cv2.imshow("original", img)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, bw_img = cv2.threshold(grey_img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary_img", bw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_sim_img()
    # get_real_img()
