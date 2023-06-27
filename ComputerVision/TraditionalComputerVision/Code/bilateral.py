import sys
import numpy as np
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/lenna.bmp")
    src = cv2.imread("./images/lenna.bmp", cv2.IMREAD_GRAYSCALE)


    if src is None:
        print("Image load failed!")
        sys.exit()

    dst = cv2.bilateralFilter(src, -1, 10, 5)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)

    cv2.waitKey()
    cv2.destroyAllWindows()