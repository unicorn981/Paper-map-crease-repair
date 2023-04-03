import cv2
import numpy as np
import matplotlib.pyplot as plt

import Global_value

Global_value._init()
filepath = Global_value.get_value('filepath')

def shadowget(ROI_cube):
    img = ROI_cube

    # 高斯滤波
    src = cv2.medianBlur(img, 3)

    # 显示图像
    # cv2.imshow("source img", img)
    # cv2.imshow("medianBlur", result)

    # # 等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 设置卷积核
    kernel1 = np.ones((10, 10), np.uint8)
    kernel2 = np.ones((10, 10), np.uint8)


    src = cv2.dilate(src, kernel1 , iterations = 1)
    result= cv2.erode(src, kernel2, iterations = 1)
    # 图像闭运算
    # result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel1)

    # 显示图像
    # cv2.imshow("src", src)
    # cv2.imshow("result", result)
    #
    # # 等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(filepath+"shadow.png", result)
    return result



def unevenLightCompensate(img, blockSize=16):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filepath+"unevenLightCompensate.png", dst)

    return dst

def Optimize(dst): #输入光度均匀处理后的图像
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # height, width = dst.shape[:2]
    # New_Img = np.zeros((height, width), dtype=np.uint8)
    # for y in range(height):
    #     for x in range(width):
    #         if 242 < gray[y, x]:
    #         # if 190 < gray_img[y, x]:
    #             New_Img[y, x] = 255
    retVal, New_Img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # img2 = cv2.imread("./image/count_gray.png")
    kernel = np.ones((3, 17), np.uint8) #以下8段代码处理基于折痕在图上横向显示，其他情况不太适用
    kernel2 = np.ones((1, 7), np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    Old_Img = New_Img
    # New_Img = cv2.morphologyEx(New_Img,cv2.MORPH_OPEN,kernel)
    # img2 = cv2.imread("./image/count_gray.png")
    New_Img = cv2.erode(New_Img, kernel2, iterations=2)
    New_Img = cv2.dilate(New_Img, kernel2, iterations=2)
    New_Img = cv2.dilate(New_Img, kernel, iterations=3)
    New_Img = cv2.erode(New_Img, kernel3, iterations=3)
    # cv2.imwrite("./image/New_Img.png",New_Img)
    return New_Img

def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)

    return mask_sel

def Rosenfeld(im):
    # im = cv2.imread("./image/binary.png", 0)

    ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    skel = np.zeros(im.shape, np.uint8)
    erode = np.zeros(im.shape, np.uint8)
    temp = np.zeros(im.shape, np.uint8)

    i = 0
    while True:
        # cv2.imshow('im %d' % (i), im)
        erode = cv2.erode(im, element)
        temp = cv2.dilate(erode, element)

        # 消失的像素是skeleton的一部分
        temp = cv2.subtract(im, temp)
        # cv2.imshow('skeleton part %d' % (i,), temp)
        skel = cv2.bitwise_or(skel, temp)
        im = erode.copy()

        if cv2.countNonZero(im) == 0:
            break
        i += 1


    skel = find_max_region(skel)
    cv2.imwrite(filepath+"output.png", skel)
    return skel

if __name__ == '__main__':
    original_img = cv2.imread(filepath+"ROI_cube.png")
    result = shadowget(original_img)
    rst = unevenLightCompensate(result)
    New_Img = Optimize(rst)
    img3 = find_max_region(New_Img)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    img4 = Rosenfeld(img3)
    cv2.imshow("result",img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

