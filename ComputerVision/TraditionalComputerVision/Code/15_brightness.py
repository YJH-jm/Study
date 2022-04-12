import sys
import cv2
import numpy as np


############## 1. gray scale ##############
src = cv2.imread("images/lenna.bmp", cv2.IMREAD_GRAYSCALE)

if src is None:
    print("Image")
# dst = src + 100 # 255보다 커지면 0이 저장되게 됨
# dst = np.clip(src+100, 0, 255) # 이것도 위와 같이 나옴

dst = cv2.add(src, 100) 
# dst = cv2.add(src, (100, 0, 0, 0))
# dst = np.clip(src+100., 0, 255).astype(np.uint8) # 위와 같은 결과가 나오는 코드
cv2.imshow("dst", dst)


############## 2. color ##############
src = cv2.imread("images/lenna.bmp")
# dst = cv2.add(src, 100) # (100, 0, 0, 0) 이므로 BGR 중 B 성분만 100 증가
dst = cv2.add(src, (100, 100, 100, 0))

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey()
cv2.destroyAllWindows()
