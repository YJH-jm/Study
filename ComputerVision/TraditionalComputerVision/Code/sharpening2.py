import sys
import numpy as np
import cv2


if __name__ == "__main__":
    src = cv2.imread("./images/rose.bmp")

    if src is None:
        print("Image load failed")
        sys.exit()

    src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
    src_f = src_ycrcb[:, :, 0].astype(np.float32) # 밝기 성분만 바꾸기 위해, 중간 연산은 flaot로 해주는 것이 더 정확
    
    blr = cv2.GaussianBlur(src_f, (0, 0), 2)
    src_ycrcb[:,:, 0]= np.clip(2.0*src_f - blr, 0, 255).astype(np.uint8) # 내부 연산은 float가 되어여

    dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    
    cv2.waitKey()
    cv2.destroyAllWindows()


