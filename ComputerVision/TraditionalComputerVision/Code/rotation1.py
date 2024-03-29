import sys
import numpy as np
import math
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/tekapo.bmp")

    if src is None:
        print("Image load failed!")
        sys.exit()

    rad = 20 * math.pi / 180
    aff = np.array([[math.cos(rad), math.sin(rad), 0], [-math.sin(rad), math.cos(rad), 0]], dtype=np.float32)
    dst = cv2.warpAffine(src, aff, (0, 0))
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    