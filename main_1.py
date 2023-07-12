import cv2
import s2_detecting3 as dt
import ill_dealing_2 as ill
import numpy as np
import Global_value
import time


# filepath = "./image6/"
# Global_value._init()
# filepath = Global_value.get_value('filepath')
def crease_detecting(original_img,shifting,filepath):
    img = original_img.copy()
    original_image = original_img.copy()
    newimg = dt.shadowget(img,filepath)
    newimg = dt.unevenLightCompensate(newimg,filepath)
    gray = dt.count_gray(newimg)
    img3 = dt.find_max_region(gray,filepath)
    cv2.imwrite(filepath + "count_gray.png", img3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    img3 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
    img3 = dt.find_max_region(img3,filepath)
    New_Img = dt.Optimize(img3,filepath)
    # cv2.imshow("result111", New_Img)
    img3 = dt.find_max_region(New_Img,filepath)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    high, low, ROI = dt.draw_convexHull(img, img3,filepath)
    img4 = dt.Rosenfeld(img3,filepath)
    dt.drawline(original_image, img4, shifting,filepath)
    return high+20,low-20

def illus_deal(src,filepath):

    light = 30
    # light = 30
    result = ill.shadow(src, light,filepath)
    cv2.imwrite(filepath + "ill_geo2.png", result)
    ill.shapal(filepath)

def geo_refic(high,low,filepath):
    original_img = cv2.imread(filepath + "ill_geo2.png")
    img = original_img.copy()
    height, width = img.shape[:2]
    crease_pos = int((high + low)/2) + 7

    img1 = img[high:crease_pos,:].copy()
    height_1 = crease_pos - high
    height1 = int(1.2*height_1)

    img2 = img[crease_pos:low, :].copy()
    height_2 = low - crease_pos
    height2 = int(1.2*height_2)


    img_1 = cv2.resize(img1, (width,height1), interpolation=cv2.INTER_CUBIC)
    img_2 = cv2.resize(img2, (width,height2), interpolation=cv2.INTER_CUBIC)
    # img_1 = t2.BiCubic_interpolation(img1,height1,width)
    # img_2 = t2.BiCubic_interpolation(img2,height2,width)

    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((height - height_2 - height_1 + height2 + height1,width,3),dtype=np.uint8)
    retimg[:high,:] = img[:high,:]
    retimg[high:high+height1,:] = img_1[:,:]
    retimg[high + height1:high + + height1 + height2, :] = img_2[:, :]
    retimg[high + height1 + height2:, :] = img[low:, :]

    cv2.imwrite(filepath + "result.png",retimg)
    # cv2.imshow("origin",img)
    # cv2.imshow("result",retimg)
    # cv2.imshow("result1", img1)
    # cv2.imshow("result_1", img_1)
    # cv2.imshow("result2",img2)
    # cv2.imshow("result_2", img_2)



if __name__ == '__main__':
    time_start = time.time()  # 开始计时
    Global_value._init()
    filepath = "./image7/"
    original_img = cv2.imread(filepath + "2222.jpg")
    img1 = original_img.copy()
    img2 = original_img.copy()

    high,low = crease_detecting(img1,shifting=-5,filepath = filepath)

    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')

    illus_deal(img2,filepath)
    geo_refic(high,low,filepath)

