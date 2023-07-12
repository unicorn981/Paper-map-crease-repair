import cv2
if __name__ == '__main__':
    img = cv2.imread("./test/img_8.png",0)
    img = 255 - img
    cv2.imwrite("./test/bin2.png",img)
