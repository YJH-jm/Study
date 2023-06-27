import sys
import numpy as np
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/tekapo.bmp")

    if src is None:
        print("Image load failed!")
        sys.exit()
    
    aff = np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32)
    dst = cv2.warpAffine(src, aff, (0,0))
    
    h, w = src.shape[:2]
    dst2 = cv2.warpAffine(src, aff, (w+ int(h*0.5), h))
    
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.imshow("dst2", dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()