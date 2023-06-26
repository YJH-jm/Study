import sys
import numpy as np
import cv2


if __name__ == "__main__":
    src = cv2.imread("./images/rose.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        sys.exit()

    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9], 
                       [1/9, 1/9, 1/9]],dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) /9.
    dst = cv2.filter2D(src, -1, kernel=kernel)
    dst2 = cv2.blur(src, ksize=(3,3))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.imshow("dsts2", dst2)

    cv2.waitKey()
    cv2.destroyAllWindows()