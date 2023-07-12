import cv2
import s2_detecting3 as dt1
import ill_dealing_2 as ill1
import s2_detecting2 as dt2
import ill_dealing_3 as ill2
import s2_detection1 as dt3
import ill_dealing as ill3

import numpy as np
import Global_value
import gradio as gr



# filepath = "./image6/"
# Global_value._init()
def crease_detecting1(original_img,shifting,filepath):
    img = original_img.copy()
    original_image = original_img.copy()
    newimg = dt1.shadowget(img,filepath)
    newimg = dt1.unevenLightCompensate(newimg,filepath)
    gray = dt1.count_gray(newimg)
    img3 = dt1.find_max_region(gray,filepath)
    cv2.imwrite(filepath + "count_gray.png", img3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    img3 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel)
    img3 = dt1.find_max_region(img3,filepath)
    New_Img = dt1.Optimize(img3,filepath)
    # cv2.imshow("result111", New_Img)
    img3 = dt1.find_max_region(New_Img,filepath)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    high, low, ROI = dt1.draw_convexHull(img, img3,filepath)
    img4 = dt1.Rosenfeld(img3,filepath)
    dt1.drawline(original_image, img4, shifting,filepath)
    return high+20,low-20 , original_image

def illus_deal1(src,filepath):

    light = 33
    # light = 30
    result = ill1.shadow(src, light,filepath)
    cv2.imwrite(filepath + "ill_geo2.png", result)
    ill1.shapal(filepath)

def geo_refic1(high,low,filepath):
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
    return retimg
    # cv2.imshow("origin",img)
    # cv2.imshow("result",retimg)
    # cv2.imshow("result1", img1)
    # cv2.imshow("result_1", img_1)
    # cv2.imshow("result2",img2)
    # cv2.imshow("result_2", img_2)



def crease_detecting2(original_img,shifting,filepath):
    img = original_img.copy()
    origin_img = img.copy()
    newimg = dt2.shadowget(img,filepath)
    # newimg = unevenLightCompensate(img)
    # newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    gray = dt2.count_gray(newimg,filepath)
    # grayimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
    # newimg = dsa.sauvola(gray)
    # cv2.imwrite('test/2_result.png',newimg)
    img3 = dt2.find_max_region(gray,filepath)
    cv2.imwrite(filepath+ "count_gray.png", img3)
    New_Img = dt2.Optimize(img3)
    # cv2.imshow("result111", New_Img)
    img3 = dt2.find_max_region(New_Img,filepath)
    cv2.imwrite(filepath + "mask_sel.png", img3)
    img4 = dt2.Rosenfeld(img3,filepath)
    high, low, ROI = dt2.draw_convexHull(origin_img, img4,filepath)
    dt2.drawline(img, img4, 6,filepath)
    # deal_ill(origin_img1, high, low)
    return high + 20 + 1,low, img

def illus_deal2(src,filepath):

    light = 30
    # light = 30
    result = ill2.shadow(src, light)
    cv2.imwrite(filepath + "ill_geo2.png", result)
    ill2.shapal(filepath)

def geo_refic2(high,low,filepath):
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
    return res



def main_1(filepath,type,image):
    # original_img = cv2.imread(filepath + "2222.jpg")
    # Global_value._init()
    # filepath = Global_value.get_value('filepath')
    # original_img = cv2.imread(filepath + "2222.jpg")
    if type == '1':
        original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(filepath) != 0:
            Global_value.set_value('filepath',filepath)
        filepath = Global_value.get_value('filepath')
        img1 = original_img.copy()
        img2 = original_img.copy()
        # cv2.imshow("nima",original_img)
        # cv2.waitKey(0)
        high,low, crease = crease_detecting1(img1,shifting=7,filepath = filepath)#7
        illus_deal1(img2,filepath)
        result = geo_refic1(high, low, filepath)
        crease_image = cv2.cvtColor(crease,cv2.COLOR_RGB2BGR)
        result_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if type == '2':
        original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(filepath) != 0:
            Global_value.set_value('filepath',filepath)
        filepath = Global_value.get_value('filepath')
        img1 = original_img.copy()
        img2 = original_img.copy()
        # cv2.imshow("nima",original_img)
        # cv2.waitKey(0)
        high,low, crease = crease_detecting2(img1,shifting=6,filepath = filepath)
        illus_deal2(img2,filepath)
        result = geo_refic2(high, low, filepath)
        crease_image = cv2.cvtColor(crease, cv2.COLOR_RGB2BGR)
        result_img = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

    if type == '3':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(filepath) != 0:
            Global_value.set_value('filepath',filepath)
        origin_img = img.copy()
        src = img.copy()
        newimg = dt3.shadowget(img, filepath)
        newimg = dt3.unevenLightCompensate(newimg, filepath)
        gray = dt3.count_gray(newimg,filepath)
        # grayimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2GRAY)
        # newimg = dsa.sauvola(gray)
        # cv2.imwrite('test/2_result.png',newimg)
        # cv2.imshow("result111", New_Img)
        img3 = dt3.find_max_region(gray)
        cv2.imwrite(filepath + "count_gray.png", img3)
        New_Img = dt3.Optimize(img3)
        # cv2.imshow("result111", New_Img)
        img3 = dt3.find_max_region(New_Img)

        cv2.imwrite(filepath + "mask_sel.png", img3)
        img4 = dt3.Rosenfeld(img3, filepath)
        high, low, ROI = dt3.draw_convexHull(origin_img, img4, filepath)
        dt3.drawline(img, img4, 7,filepath)
        crease_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        light = 28
        # light = 30
        result = ill3.shadow(src, light)
        cv2.imwrite(filepath + "ill_geo.png", result)
        result = ill3.shapal(filepath)
        result_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return crease_image,result_img

Global_value._init()
filepath = Global_value.get_value('filepath')
# main_1(cv2.imread(filepath + "2222.jpg"))
interface = gr.Interface(fn=main_1, inputs=[gr.inputs.Textbox(lines=1, placeholder="输入路径",label="中间结果路径"),
                                            gr.inputs.Textbox(lines=1, placeholder="1为凸折痕，2为凹折痕",label="折痕类型"),
                                            gr.inputs.Image(label="输入图像")],
                                            outputs=[gr.Image(label="折痕检测结果"),
                                                     gr.Image(label="扭曲修复结果")])
interface.launch()