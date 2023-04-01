import cv2
import numpy as np
from matplotlib import pyplot as plt

# 方法：大津法
gray = cv2.imread('./image/ROI_cube.png', 0)
_, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("./test_out1.jpg", gray)


