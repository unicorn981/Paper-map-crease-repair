import cv2
import numpy as np
import ill_dealing
from skimage import morphology
from matplotlib import pyplot as plt
def calcGrayHist(grayimage):
    # 灰度图像矩阵的高，宽
    rows, cols = grayimage.shape

    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[grayimage[r][c]] += 1

    return grayHist


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
    for k in range(255):
        top += histogram[k]
        if (top/total)>=0.04:
            break
    peak_set = k


    thresh = peak_set
    # 找到阈值之后进行阈值处理，得到二值图

    return thresh

def unevenLightCompensate(img, filepath, blockSize=16):
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
    cv2.imwrite(filepath + "unevenLightCompensate.png", dst)

    return dst

def shadowget(ROI_cube,filepath):
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
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)


    src = cv2.dilate(src, kernel1 , iterations = 6)
    result= cv2.erode(src, kernel2, iterations = 6)
    # 图像闭运算
    # result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel1)

    # 显示图像
    # cv2.imshow("src", src)
    # cv2.imshow("result", result)
    #
    # # 等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(filepath + "shadow.png", result)
    return result


def count_gray(original_img,filepath):
    height, width = original_img.shape[:2]
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1",gray_img)
    # cv2.waitKey(0)
    num_color = gray_img.reshape(1, -1)[0]
    len_color = int(len(num_color) * (1 - 0.05))
    print("[num_color]", len(num_color))
    print("[len_color]", len_color)

    New_Img = np.zeros((height, width), dtype=np.uint8)
    thre = threshTwoPeaks(original_img)
    for y in range(height):
        for x in range(width):

            # if 242 < gray_img[y, x]:
            if thre > gray_img[y, x]:
                New_Img[y, x] = 255
    # ret ,New_Img = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    cv2.imwrite(filepath + "count_gray.png", New_Img)
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

def Optimize(dst): #输入二值化折痕后图像
    New_Img = dst.copy()
    # height, width = dst.shape[:2]
    # New_Img = np.zeros((height, width), dtype=np.uint8)
    # for y in range(height):
    #     for x in range(width):
    #         if 242 < gray[y, x]:
    #         # if 190 < gray_img[y, x]:
    #             New_Img[y, x] = 255
    # cv2.imshow("1s",gray)
    # retVal, New_Img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # New_Img = count_gray(dst)
    # cv2.imshow("s",New_Img)
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
    # New_Img = cv2.erode(New_Img, kernel, iterations=3)
    New_Img = cv2.erode(New_Img, kernel3, iterations=2)
    # New_Img = cv2.dilate(New_Img, kernel3, iterations=1)
    # cv2.imwrite("./image/New_Img.png",New_Img)
    return New_Img

def Rosenfeld(im,filepath):
    # im = cv2.imread("./image/binary.png", 0)
    binary = im.copy()

    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    cv2.imwrite(filepath + "dist_on_skel.png", dist_on_skel)
    return dist_on_skel

    # ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #
    # skel = np.zeros(im.shape, np.uint8)
    # erode = np.zeros(im.shape, np.uint8)
    # temp = np.zeros(im.shape, np.uint8)
    #
    # i = 0
    # while True:
    #     # cv2.imshow('im %d' % (i), im)
    #     erode = cv2.erode(im, element)
    #     temp = cv2.dilate(erode, element)
    #
    #     # 消失的像素是skeleton的一部分
    #     temp = cv2.subtract(im, temp)
    #     # cv2.imshow('skeleton part %d' % (i,), temp)
    #     skel = cv2.bitwise_or(skel, temp)
    #     im = erode.copy()
    #
    #     if cv2.countNonZero(im) == 0:
    #         break
    #     i += 1
    #
    #
    # # skel = find_max_region(skel)
    # cv2.imwrite(filepath + "output.png", skel)
    # return skel

def draw_convexHull(original_img, New_Img, filepath):
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
            cv2.imwrite(filepath + "ROI.png", original_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('roi',ROI)
            ROI = original_img[contours_ok[1] - 10:contours_ok[3] + 10,:]
            cv2.imwrite(filepath + "ROI_cube.png", ROI)
            high = contours_ok[1] - 10
            low = contours_ok[3] + 10
            return high,low,ROI



def drawline(img,crease,shifting,filepath):
    rows, cols = crease.shape[:2]

    # 存储灰度直方图
    for y in range(rows):
        for x in range(cols):
            if 0 < crease[y, x] and y+shifting < rows:
                img[y+shifting,x] = [0,255,0]
    # cv2.imshow("result", img)
    cv2.imwrite(filepath + "crease.png",img)
    return img
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    filepath = "./image3/"
    img = cv2.imread("./image3/2222.jpg")
    origin_img = img.copy()
    origin_img1 = img.copy()
    newimg = shadowget(img,filepath)
    newimg = unevenLightCompensate(newimg,filepath)
    gray = count_gray(newimg,filepath)
    # grayimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    # newimg = dsa.sauvola(gray)
    # cv2.imwrite('test/2_result.png',newimg)
    # cv2.imshow("result111", New_Img)
    img3 = find_max_region(gray)
    cv2.imwrite(filepath + "count_gray.png",img3)
    New_Img = Optimize(img3)
    # cv2.imshow("result111", New_Img)
    img3 = find_max_region(New_Img)

    cv2.imwrite(filepath + "mask_sel.png", img3)
    img4 = Rosenfeld(img3,filepath)
    high, low, ROI = draw_convexHull(origin_img, img4,filepath)
    drawline(img, img4,7,filepath)
    # deal_ill(origin_img1, high, low)




