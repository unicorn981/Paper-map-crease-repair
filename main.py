import s1_ROI_obtain as roi
import s2_morphology_processing as mp
import cv2
import Global_value



Global_value._init()
filepath = Global_value.get_value('filepath')

def ROIget(original_img):
    roi.show_hist(original_img)
    # roi.count_colors(original_img)
    New_Img = roi.count_gray(original_img)
    high , low , ROI= roi.draw_convexHull(original_img, New_Img)
    return high,low,ROI

def mor_proc(ROI):
    result = mp.shadowget(ROI)
    rst = mp.unevenLightCompensate(result)
    New_Img = mp.Optimize(rst)
    img3 = mp.find_max_region(New_Img)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    img4 = mp.Rosenfeld(img3)
    # cv2.imshow("result", img4)
    return img4

def drawline(img,crease,high , low):
    rows, cols = crease.shape[:2]

    # 存储灰度直方图
    for y in range(rows):
        for x in range(cols):
            if 0 < crease[y, x]:
                img[high+y,x] = [0,155,0]
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    original_img = cv2.imread(filepath+"2222.jpg")
    imgfor = original_img.copy()
    high , low , ROI = ROIget(imgfor)
    img3 = mor_proc(ROI)

    drawline(original_img,img3,high-5,low)
    # cv2.imshow("result",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
