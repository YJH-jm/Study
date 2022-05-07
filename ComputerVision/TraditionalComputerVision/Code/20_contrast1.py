import cv2
import sys
import numpy as np

if __name__ == "__main__":
    src = cv2.imread("images/lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        sys.exit()

    alpha = 1.0 # contrast 올리기/

    dst = np.clip((1 + alpha) * src - 128 * alpha, 0, 255).astype(np.uint8)
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()

    cv2.destroyAllWindows()