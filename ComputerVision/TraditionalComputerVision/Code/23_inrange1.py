import sys
import numpy as np
import cv2

# src = cv2.imread("images/candies.png")
src = cv2.imread("images/candies2.png") # 조금 더 어둡게 한 사진

if src is None:
    print("image load failed")
    sys.exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100))
dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255))

cv2.imshow("src", src)
cv2.imshow("dst1", dst1)
cv2.imshow("dsr2", dst2)

cv2.waitKey()
cv2.destroyAllWindows()