import sys
import numpy as np
import cv2

if __name__ == "__main__":
    src = cv2.imread("./images/tekapo.bmp")

    if src is None:
        print("Image load failed!")
        sys.exit()

    # 가로로 200 pixel, 세로로 100 pixel 이동 
    aff = np.array([[1, 0, 200],[0, 1, 100]],dtype=np.float32)
    dst = cv2.warpAffine(src, aff, (0,0))
    
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()