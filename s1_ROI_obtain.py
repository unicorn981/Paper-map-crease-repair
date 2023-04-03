import cv2
import numpy as np
import matplotlib.pyplot as plt
import s1_1_hist as hi

import Global_value

Global_value._init()
filepath = Global_value.get_value('filepath')

def calcAndDrawHist(image):
    return cv2.calcHist([image], [0], None, [256], [0.0, 255.0])


def show_hist(original_img):
    b, g, r = cv2.split(original_img)
    histImgB = calcAndDrawHist(b)
    histImgG = calcAndDrawHist(g)
    histImgR = calcAndDrawHist(r)
    plt.plot(histImgB, 'b')
    plt.plot(histImgG, 'g')
    plt.plot(histImgR, 'r')
    plt.show()


def count_colors(original_img):  # 测试用
    height, width = original_img.shape[:2]
    b, g, r = cv2.split(original_img)
    b_color = np.multiply(b, 65025)
    g_color = np.multiply(g, 250)
    colors = b_color + g_color + r
    num_color = colors.reshape(1, -1)[0]
    len_color = int(len(num_color) * (1 - 0.05))
    print("[num_color]", len(num_color))
    print("[len_color]", len_color)

    histImg = np.bincount(num_color)
    New_Img = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if 55955 < colors[y, x]:
                New_Img[y, x] = 255
    plt.plot(histImg, 'b')
    plt.show()


def count_gray(original_img):
    height, width = original_img.shape[:2]
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1",gray_img)
    # cv2.waitKey(0)
    num_color = gray_img.reshape(1, -1)[0]
    len_color = int(len(num_color) * (1 - 0.05))
    print("[num_color]", len(num_color))
    print("[len_color]", len_color)

    New_Img = np.zeros((height, width), dtype=np.uint8)
    thre = hi.threshTwoPeaks(original_img)
    for y in range(height):
        for x in range(width):

            # if 242 < gray_img[y, x]:
            if thre < gray_img[y, x]:
                New_Img[y, x] = 255
    # ret ,New_Img = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    cv2.imwrite(filepath+"count_gray.png", New_Img)
    return New_Img


def draw_convexHull(original_img, New_Img):
    height, width = original_img.shape[:2]
    ret, thresh = cv2.threshold(New_Img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_ok = []
    len_contours = len(contours)
    if 0 < len_contours:
        contours_temp = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 5 and h >= 3:
                contours_temp.append([x, y, x + w, y + h])
        if 0 < len(contours_temp):
            contours_np = np.array(contours_temp)
            col_max = np.max(contours_np, axis=0)
            col_min = np.min(contours_np, axis=0)
            contours_ok = [col_min[0], col_min[1], col_max[2], col_max[3]]
            print("contours_ok", contours_ok)
            cv2.rectangle(original_img, (0, contours_ok[1] - 20), (width, contours_ok[3] + 20), (255, 255, 255), 4)
            # cv2.imshow('line', original_img)
            cv2.imwrite(filepath+"ROI.png", original_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('roi',ROI)
            ROI = original_img[contours_ok[1] - 10:contours_ok[3] + 10,:]
            cv2.imwrite(filepath+"ROI_cube.png", ROI)
            high = contours_ok[1] - 10
            low = contours_ok[3] + 10
            return high,low,ROI


if __name__ == '__main__':
    original_img = cv2.imread(filepath+"2222.jpg")
    show_hist(original_img)
    count_colors(original_img)
    New_Img = count_gray(original_img)
    draw_convexHull(original_img, New_Img)
