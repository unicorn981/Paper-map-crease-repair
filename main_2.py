import cv2
import s2_detecting2 as dt
import ill_dealing_3 as ill
import numpy as np
import Global_value
import time
# filepath = "./image4/"
filepath = "./image7/"

def crease_detecting(original_img,shifting):
    img = original_img.copy()
    origin_img = img.copy()
    newimg = dt.shadowget(img,filepath)
    # newimg = unevenLightCompensate(img)
    # newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    gray = dt.count_gray(newimg,filepath)
    # grayimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    # newimg = dsa.sauvola(gray)
    # cv2.imwrite('test/2_result.png',newimg)
    img3 = dt.find_max_region(gray,filepath)
    cv2.imwrite(filepath+ "count_gray.png", img3)
    New_Img = dt.Optimize(img3)
    # cv2.imshow("result111", New_Img)
    img3 = dt.find_max_region(New_Img,filepath)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    img4 = dt.Rosenfeld(img3,filepath)
    high, low, ROI = dt.draw_convexHull(origin_img, img4,filepath)
    dt.drawline(img, img4, -5,filepath)
    # deal_ill(origin_img1, high, low)
    return high + 20 + 1,low

def illus_deal(src):

    light = 30
    # light = 30
    result = ill.shadow(src, light)
    cv2.imwrite(filepath + "ill_geo2.png", result)
    ill.shapal(filepath)

def geo_refic(high,low):
    original_img = cv2.imread(filepath + "ill_geo2.png")
    img = original_img.copy()
    height, width = img.shape[:2]

    crease_pos = 490

    img1 = img[high:crease_pos, :].copy()
    height_1 = crease_pos - high
    height1_1 = int(height_1 / 2)
    img11 = img1[:height1_1, :]
    height1_1 = int(1.3 * height1_1)

    height1_2 = int(height_1 / 2)
    img12 = img1[height1_2:, :]
    height1_2 = int(2 * height1_2)
    img1_1 = cv2.resize(img11, (width, height1_1), interpolation=cv2.INTER_CUBIC)
    img1_2 = cv2.resize(img12, (width, height1_2), interpolation=cv2.INTER_CUBIC)
    height1 = height1_1 + height1_2
    img_1 = np.zeros((height1, width, 3), dtype=np.uint8)
    img_1[:height1_1, :] = img1_1[:, :]
    img_1[height1_1:, :] = img1_2[:, :]

    img2 = img[crease_pos:low, :].copy()
    height_2 = low - crease_pos
    height2_1 = int(height_2 / 2)
    img21 = img2[:height2_1, :]
    height2_1 = int(1.4 * height2_1)

    height2_2 = int(height_2 / 2)
    img22 = img2[height2_2:, :]
    height2_2 = int(1.3 * height2_2)
    img2_1 = cv2.resize(img21, (width, height2_1), interpolation=cv2.INTER_CUBIC)
    img2_2 = cv2.resize(img22, (width, height2_2), interpolation=cv2.INTER_CUBIC)
    # img_1 = t2.BiCubic_interpolation(img1,height1,width)
    # img_2 = t2.BiCubic_interpolation(img2,height2,width)
    height2 = height2_1 + height2_2
    img_2 = np.zeros((height2, width, 3), dtype=np.uint8)
    img_2[:height2_1, :] = img2_1[:, :]
    img_2[height2_1:, :] = img2_2[:, :]

    scrH, scrW, _ = img.shape
    # img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg = np.zeros((height - height_2 - height_1 + height2 + height1, width, 3), dtype=np.uint8)
    retimg[:high, :] = img[:high, :]
    retimg[high:high + height1, :] = img_1[:, :]
    retimg[high + height1:high + height1 + height2, :] = img_2[:, :]
    retimg[high + height1 + height2:, :] = img[low:, :]

    maskimg = np.zeros((height - height_2 - height_1 + height2 + height1, width), dtype=np.uint8)
    maskimg[high + height1 - 1:high + height1, :] = 255
    res = cv2.inpaint(src=retimg, inpaintMask=maskimg, inpaintRadius=3, flags=cv2.INPAINT_NS)
    cv2.imwrite(filepath + "result.png",res)

if __name__ == '__main__':
    original_img = cv2.imread(filepath + "2222.jpg")
    img1 = original_img.copy()
    img2 = original_img.copy()
    time_start = time.time()  # 开始计时
    high,low = crease_detecting(img1,shifting=6)
    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')

    illus_deal(img2)
    geo_refic(high,low)

