import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

    print('双峰为：',peak_set)

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值

    thresh = peak_set + 2
    # 找到阈值之后进行阈值处理，得到二值图
    threshImage_out = gray.copy()
    # 大于阈值的都设置为255
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0
    return thresh, threshImage_out

if __name__ == "__main__":

    img = cv.imread('./image/2222.jpg')
    #img = cv.imread('2.png')
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    #灰度直方图曲线
    hist = cv.calcHist([img_gray], [0], None, [256], [0, 255]) #对图像像素的统计分布，它统计了每个像素（0到L-1）的数量。
    plt.plot(hist)
    plt.show()

    #灰度直方图
    plt.hist(img_gray.ravel(), 256), plt.title('hist') #ravel()方法将数组维度拉成一维数组
    # plt.show()
    plt.savefig('1_hist.png')

    thresh, img_sep = threshTwoPeaks(img)
    print('灰度阈值为:',thresh)

    cv.imwrite('1_sep.png', img_sep)
    # cv.imwrite('2_sep.png', img_sep)


