import cv2
import numpy as np
import matplotlib.pyplot as plt

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


    # src = cv2.dilate(src, kernel1 , iterations = 8)
    # src = cv2.erode(src, kernel2, iterations = 4)
    # 图像闭运算
    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel1)

    # 显示图像
    # cv2.imshow("src", src)
    # cv2.imshow("result", result)
    #
    # # 等待显示
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("./image/shadow.png", result)
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
    cv2.imwrite("./image/unevenLightCompensate.png", dst)

    return dst



if __name__ == '__main__':
    original_img = cv2.imread("./image/ROI_cube.png")
    result = shadowget(original_img)
    rst = unevenLightCompensate(result)
    cv2.imshow("result",rst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

