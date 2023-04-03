import cv2
import numpy as np

# 计算灰度直方图
def calcGrayHist(grayimage):
    # 灰度图像矩阵的高，宽
    rows, cols = grayimage.shape

    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[grayimage[r][c]] += 1

    return grayHist

# 阈值分割：直方图技术法
def threshTwoPeaks(image):

    #转换为灰度图
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度直方图
    histogram = calcGrayHist(gray)
    # 寻找灰度直方图的最大峰值对应的灰度值
    # for k in range(1,255):
    #     if histogram[k] > histogram[k-1] and histogram[k] > histogram[k+1]:
    peak_set = 0
    # print(maxLoc)
    total = gray.size
    top = 0
    for k in range(254,0,-1):
        top += histogram[k]
        if (top/total)>=0.005:
            break
    peak_set = k


    thresh = peak_set + 2
    # 找到阈值之后进行阈值处理，得到二值图

    return thresh