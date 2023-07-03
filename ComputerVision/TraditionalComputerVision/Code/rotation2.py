import sys
import numpy as np
import math
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/tekapo.bmp")

    if src is None:
        print("Image load failed!")
        sys.exit()

    cp = (src.shape[1]/2, src.shape[0]/2) # (가로 크기의 반, )
    rot = cv2.getRotationMatrix2D(cp, 20, 1)
    dst = cv2.warpAffine(src, rot, (0, 0))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    