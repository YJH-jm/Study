import cv2
import sys
import numpy as np


def on_trackbar(pos):
    # 2개의 값을 받아야하기 때문에 pos 라는 함수 대신 getTrackbarPos 라는 함수 사용
    hmin = cv2.getTrackbarPos("H_min", "dst")
    hmax = cv2.getTrackbarPos("H_max", "dst")

    dst = cv2.inRange(src_hsv, (hmin, 150, 0), (hmax, 255, 255))
    cv2.imshow("dst", dst)

    print(hmin, hmax)

if __name__ == "__main__":
    src = cv2.imread("images/candies.png")

    if src is None:
        print("Image load failed!")

    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    cv2.imshow("src", src)
    cv2.namedWindow("dst")

    cv2.createTrackbar("H_min", "dst", 50, 179, on_trackbar) # H 값의 lowerbound
    cv2.createTrackbar("H_max", "dst", 80, 179, on_trackbar) # H 값의 upperbound

    on_trackbar(0)

    cv2.waitKey()
    cv2.destroyAllWindows()
