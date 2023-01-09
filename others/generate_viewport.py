import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
pi = np.pi


def bicubic_interpoaion(im, x_out, y_out, width, height):
    """双三次插值"""
    y_f = int(y_out)
    x_f = int(x_out)
    p = y_out - y_f
    q = x_out - x_f
    if y_f == 0:
        p = 0
    if y_f >= height - 1:
        y_f = height - 1
        print(1, y_f, np.mod(x_f - 1, width), y_f, np.mod(x_f, width))
        return (1 - q) * im[y_f, np.mod(x_f - 1, width) + 1] + q * im[y_f, np.mod(x_f, width) + 1]
    else:
        # print(2, [y_f, np.mod(x_f - 1, width)], [y_f, np.mod(x_f, width)], [y_f + 1, np.mod(x_f - 1, width)],
        # [y_f + 1, np.mod(x_f, width)])
        """return (1 - p) * (1 - q) * im[y_f, np.mod(x_f - 1, width)] + (1 - p) * q * im[y_f, np.mod(x_f, width)] \
               + p * (1 - q) * im[y_f + 1, np.mod(x_f - 1, width)] + p * q * im[y_f + 1, np.mod(x_f, width)]"""
        return ((1 - p) * (1 - q) * im[y_f, np.mod(x_f, width)]
                + (1 - p) * q * im[y_f, np.mod(x_f + 1, width)]
                + p * (1 - q) * im[y_f + 1, np.mod(x_f, width)]
                + p * q * im[y_f + 1, np.mod(x_f + 1, width)])


def viewport_generation(im, beta, a, FOV):  # 0 <= beta <= 2pi, -pi <= a <= pi
    """vr图像转平面图像
    im: vr图像 beta: 球体左转角度 a: 球体上转角度 FOV: 视场角
    """
    height = im.shape[0]
    width = im.shape[1]
    # 4:3 field of view
    F_h = FOV
    F_v = 0.75 * FOV
    # 区域画面大小。经实验，最终画面显示范围与该尺寸无关。且尺寸过小，画面分辨率过低，尺寸过大；画面效果饱和，消耗时间过多
    viewport_width = np.floor(F_h / (2 * pi) * width)
    viewport_height = np.floor(F_v / pi * height)
    # viewport_height = 0.1 * height
    # viewport_width = 0.1 * width
    # print("height:", height)
    # print("width:", width)
    # print("viewport_height:", viewport_height)
    # print("viewport_width:", viewport_width)
    # print(viewport_width, viewport_height, width, height)
    
    viewport = np.zeros([int(viewport_height), int(viewport_width), 3])   # 创建一个空白的局部平面图
    # 右手笛卡尔坐标系下由欧拉角表示的旋转矩阵
    R = np.array([[np.cos(beta), np.sin(beta) * np.sin(a), np.sin(beta) * np.cos(a)],
                  [0, np.cos(a), -np.sin(a)],
                  [-np.sin(beta), np.cos(beta) * np.sin(a), np.cos(beta) * np.cos(a)]])
    
    for i in range(int(viewport_height) - 1):
        for j in range(int(viewport_width) - 1):
            u = (j + 0.5) * 2 * np.tan(F_h / 2) / viewport_width
            v = (i + 0.5) * 2 * np.tan(F_v / 2) / viewport_height

            # 由平面投影至球面
            x1 = u - np.tan(F_h / 2)
            y1 = -v + np.tan(F_v / 2)
            z1 = 1.0

            r = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
            sphere_coords = [x1 / r, y1 / r, z1 / r]    # 单位化的球面坐标
            
            # 对该点进行球面旋转操作
            rotated_sphere_coords = np.matmul(R, sphere_coords) # matmul自适应矩阵相乘：(3,3) × (1,3) -> (3,3) × (3,1)

            x = rotated_sphere_coords[0]
            y = rotated_sphere_coords[1]
            z = rotated_sphere_coords[2]
            
            # 由直角坐标系转换成球坐标系
            theta = np.arccos(y)
            phi = np.arctan2(x, z)
            # 反向投影，由球面投影至平面
            x_out = width * phi / (2 * np.pi)
            y_out = height * theta / np.pi
            # print(x_out, y_out)
            viewport[i, j] = bicubic_interpoaion(im, x_out, y_out, width, height)
            # if viewport[i, j, 0] >= 255 or viewport[i, j, 1] >= 255 or viewport[i, j, 2] >= 255:
            #     print(viewport[i, j])
            #     print(i, j)
            #     time.sleep(1)

    # viewport[:, :, 0] = viewport[:, :, 0] / viewport[:, :, 0].max() - viewport[:, :, 0].min()
    # viewport[:, :, 1] = viewport[:, :, 1] / viewport[:, :, 1].max() - viewport[:, :, 1].min()
    # viewport[:, :, 2] = viewport[:, :, 2] / viewport[:, :, 2].max() - viewport[:, :, 2].min()
    # print(viewport[242, 298])
    # tmp = viewport
    viewport = (viewport - viewport.min()) / (viewport.max() - viewport.min())
    viewport = viewport * 255
    viewport = viewport.astype(np.uint8)

    return viewport


if __name__ == '__main__':
    vr_img = plt.imread('./data/vr.jpg')
    start = time.time()
    flat_img = viewport_generation(vr_img, pi, 0, FOV=0.25 * pi)
    end =time.time()
    print("cost time:", end-start)
    
    plt.figure(figsize=(11,6))
    plt.subplot(121)
    plt.imshow(vr_img)
    plt.title("vr图像")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(flat_img)
    plt.title("vr图像转平面图像")
    plt.axis("off")
    plt.show()