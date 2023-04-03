import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def GetContours(img,Get_cont):
    contours, hierarchy = cv2.findContours(Get_cont, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    dst = img.copy()
    if w >h :
        dst = cv2.rectangle(dst, (x, y-30), (x + w, y + h+30), (255, 255, 255), 3)
        cv2.imshow('result', dst)
        cv2.waitKey(0)

        img4 = img[y - 30:y + h + 30,x:x + w ]
        cv2.imshow('4', img4)
        cv2.waitKey(0)
    else:
        dst = cv2.rectangle(dst, (x-30, y), (x + w+30, y + h), (255, 255, 255), 3)
        cv2.imshow('result', dst)
        cv2.waitKey(0)

        img4 = img[y:y + h, x - 30:x + w + 30]
        cv2.imshow('4', img4)
        cv2.waitKey(0)

    return img4

if __name__ == '__main__':
    dst = cv2.imread("./image/unevenLightCompensate.png")
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # height, width = dst.shape[:2]
    # New_Img = np.zeros((height, width), dtype=np.uint8)
    # for y in range(height):
    #     for x in range(width):
    #         if 242 < gray[y, x]:
    #         # if 190 < gray_img[y, x]:
    #             New_Img[y, x] = 255
    retVal, New_Img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("dsa",New_Img)
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
    # cv2.imshow('result1', New_Img)
    img3 = find_max_region(New_Img)


    #感兴趣区域提取
    # result = GetContours(img3)
    cv2.imshow('result', img3)
    cv2.waitKey(0)

