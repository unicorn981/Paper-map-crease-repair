import ROI_obtain as roi
import morphology_processing as mp
import cv2


def ROIget(original_img):
    roi.show_hist(original_img)
    # roi.count_colors(original_img)
    New_Img = roi.count_gray(original_img)
    high , low , ROI= roi.draw_convexHull(original_img, New_Img)
    return high,low,ROI

def mor_proc(ROI):
    mp.shadowget(ROI)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    original_img = cv2.imread("./image/2222.jpg")
    high , low , ROI = ROIget(original_img)
    mor_proc(ROI)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
