import math
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np

def shadow(input, light):
    # 生成灰度图
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    height, width = input.shape[:2]

    # 确定阴影区
    thresh = (1.0 - gray) * (1.0 - gray)

    # 取平均值作为阈值
    t = np.mean(thresh)
    # mask = cv2.imread(filepath + "mask_sel.png",0)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for i in range(height):
    # for i in range(459,505):
        for j in range(width):
            if thresh[i,j] < t/3 : mask[i,j] = 0
            elif thresh[i,j] >= t/3 and thresh[i,j] <t : mask[i,j] = 127

    mask[thresh >= t] = 255


    # cv2.imshow("ie",mask)
    # cv2.waitKey()
    # 参数设置
    max_val = 4
    bright = light / 100.0 / max_val
    mid = 1.0 + max_val * bright

    # 边缘平滑过渡
    midrate = np.zeros((height, width), dtype=np.float32)
    brightrate = np.zeros((height, width), dtype=np.float32)

    # for i in range(459,505):
    for i in range(height):
        for j in range(input.shape[1]):
            if mask[i, j] == 255:
                midrate[i, j] = mid
                brightrate[i, j] = bright
            elif mask[i,j] == 127:
                midrate[i, j] = ((mid - 1.0) / t) * thresh[i, j] + 1.0
                brightrate[i, j] = (1.0 / t) * thresh[i, j] * bright
            else:
                midrate[i, j] = 1.0
                brightrate[i, j] = 0

    # 阴影提亮，获取结果图
    result = input.copy()
    for i in range(height):
    # for i in range(459,505):
        for j in range(input.shape[1]):
            for k in range(3):
                inp = input[i,j,k]
                mm = midrate[i,j]
                bb = brightrate[i,j]
                temp = pow(float(inp) / 255.0, 1.0 / mm)*(1.0 / (1 - bb))
                temp = min(1.0, max(0.0, temp))
                temp = temp * 255.0
                result[i, j, k] = temp

    return result

def shapal(filepath):
    src = cv2.imread(filepath + "ill_geo.png")
    # cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    # cv2.imshow("input", src)

    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    # usm = cv2.addWeighted(usm, 1.5, blur_img, -0.5, 0)
    # cv2.imshow("mask image", usm)

    h, w = src.shape[:2]
    result = np.zeros([h, w * 2, 3], dtype=src.dtype)
    result[0:h, 0:w, :] = src
    result[0:h, w:2 * w, :] = usm
    # cv2.putText(result, "original image", (10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv2.putText(result, "sharpen image", (w + 10, 30), cv2.FONT_ITALIC, 1.0, (0, 0, 255), 2)
    # cv2.imshow("sharpen_image", result)
    cv2.imwrite(filepath + "illu_rec.png", usm)
    return usm



if __name__ == '__main__':
    filepath = "./image3/"
    src = cv2.imread(filepath + "2222.jpg")
    light = 30
    # light = 30
    result = shadow(src, light)
    cv2.imwrite(filepath + "ill_geo.png",result)
    shapal(filepath)
    # 显示原始图像和结果图像
    # cv2.imshow("original", src)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


