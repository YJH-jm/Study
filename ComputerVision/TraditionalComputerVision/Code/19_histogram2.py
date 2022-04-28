import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2




def getGrayHistImage(hist):
    imgHist = np.full((100, 256), 255, dtype=np.uint8) # 세로 100, 가로 256 작은 크기의 영상 흰색으로 만듦

    histMax = np.max(hist) # 히스토그램의 최댓값을 미리 계산
    print(np.max(hist))
    for x in range(256): # 밑에서 부터 하나씩 올려서 그려줌
        # print(x)
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax)) # pt1과 같은 x 좌표에서 위에 있는 부분
        # print(hist[x, 0]) 
        cv2.line(imgHist, pt1, pt2, 0)

    return imgHist

src = cv2.imread("images/lenna.bmp", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image load failed!")
    sys.exit()

hist = cv2.calcHist([src], [0], None, [256], [0, 256])

# print(hist.shape) # (256, 1)

histImg = getGrayHistImage(hist)

cv2.imshow("src", src)
cv2.imshow("histImg", histImg)
cv2.waitKey()
cv2.destroyAllWindows()