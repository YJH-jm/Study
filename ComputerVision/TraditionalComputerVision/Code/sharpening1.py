import sys
import numpy as np
import cv2


if __name__ == "__main__":
    src = cv2.imread("./images/rose.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed")
        sys.exit()

    blr = cv2.GaussianBlur(src, (0, 0), 2)
    # dst = cv2.subtract(src, blr) # 음수 부분은 subtract 함수가 saturation 시켜 0으로 만듦
    diff = cv2.addWeighted(src, 1, blr, -1, 128)
    dst =  cv2.addWeighted(src, 2, blr, -1, 0)
    dst2 = np.clip(2.0*src - blr, 0, 255).astype(np.uint8) # 내부 연산은 float가 되어여

    cv2.imshow("src", src)
    cv2.imshow("blr", blr)
    cv2.imshow("dst", dst)
    cv2.imshow("dst2", dst2)

    
    cv2.waitKey()
    cv2.destroyAllWindows()


