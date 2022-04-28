import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 그래이스케일 영상의 히스토그램
src = cv2.imread("images/lenna.bmp", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

hist = cv2.calcHist([src], [0], None, [256], [0, 256])
print("hist : ", hist)
print("hist 길이 : ", len(hist)) # hist 길이 :  256
cv2.imshow("src", src)
cv2.waitKey(1)

plt.plot(hist)
plt.show()

# 컬러 영상의 히스토그램
src = cv2.imread("images/lenna.bmp")

if src is None:
    print("Image load failed!")
    sys.exit()

colors = ["b","g", "r"]
bgr_planes = cv2.split(src)

cv2.imshow("src", src)
cv2.waitKey(1)

for (p, c) in zip(bgr_planes, colors):
    hist = cv2.calcHist([p], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)

plt.show()